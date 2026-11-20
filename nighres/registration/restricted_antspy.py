# basic dependencies
import os
import sys
from glob import glob
import math

# main dependencies: numpy, nibabel, ants
import numpy
import nibabel
import ants.utils
import scipy.ndimage

# nighresjava and nighres functions
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir
from ..surface import probability_to_levelset
from . import embedded_antspy_2d_multi, apply_coordinate_mappings

# convenience labels
X=0
Y=1
Z=2
T=3

def restricted_antspy(source_images, target_images, 
                    axis=0, depth=2,
                    direction='center',
                    run_rigid=True,
                    rigid_iterations=1000,
                    run_similarity=False,
                    similarity_iterations=0,
                    run_affine=False,
                    affine_iterations=0,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, 
                    fine_iterations=40,
					scaling_factor=8,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False, smooth_mask=0.0,
					save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Restricted ANTSpy Registration

    Runs the Symmetric Normalization (SyN) algorithm of ANTs in a restricted 2.5D approach
    along the chosen axis. 2D registration is performed from the center outward following a
    Markov chain method to ensure 3D smoothness.
    Note that source and target images must be in the same 3D space (same dims and res)

    Parameters
    ----------
    source_images: [niimg]
        Image list to register
    target_images: [niimg]
        Reference image list to match
    axis: int
        The axis along which to restrict the registration (default is 0)
    depth: int    
        The  number of images to include in the Markov chain (default is 2)
    direction: str
        Directions for the computation {'center','symmetric','sweep'}
    run_rigid: bool
        Whether or not to run a rigid registration first (default is True)
    rigid_iterations: float
        Number of iterations in the rigid step (default is 1000)
    run_similarity: bool
        Whether or not to run a similarity (rigid+scale) registration first (default is False)
    similarity_iterations: float
        Number of iterations in the similarity step (default is 1000)
    run_affine: bool
        Whether or not to run a affine registration first (default is False)
    affine_iterations: float
        Number of iterations in the affine step (default is 0)
    run_syn: bool
        Whether or not to run a SyN registration (default is True)
    coarse_iterations: float
        Number of iterations at the coarse level (default is 40)
    medium_iterations: float
        Number of iterations at the medium level (default is 50)
    fine_iterations: float
        Number of iterations at the fine level (default is 40)
    scaling_factor: int
        Number of spatial scales to use the SyN image pyramid (default is 8)
    cost_function: {'CrossCorrelation', 'MutualInformation'}
        Cost function for the registration (default is 'MutualInformation')
    interpolation: {'NearestNeighbor', 'Linear'}
        Interpolation for the registration result (default is 'NearestNeighbor')
    regularization: {'Low', 'Medium', 'High'}
        Regularization preset for the SyN deformation (default is 'Medium')
    convergence: float
        Threshold for convergence, can make the algorithm very slow (default is 1e-6)
    mask_zero: bool
        Mask regions with zero value using ANTs masking option (default is False)
    smooth_mask: float
        Smoothly mask regions within a given ratio of the object's thickness,
        in [0.0, 1.0] (default is 0.0). This does not use ANTs masking.
    save_data: bool
        Save output data to file (default is False)
    overwrite: bool
        Overwrite existing results (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * transformed_sources ([niimg]): Deformed focused source image list (_rarg_def0,1,...)
        * mapping (niimg): Coordinate mapping from source to target (_rarg_map)
        * inverse (niimg): Inverse coordinate mapping from target to source (_rarg_invmap)

    Notes
    ----------
    The SyN algorithm is part of the ANTs software by Brian Avants and colleagues [1]_. 
    Parameters have been set to values commonly found in neuroimaging scripts online, but not 
    necessarily optimal.

    References
    ----------
    .. [1] Avants et al (2008), Symmetric diffeomorphic
       image registration with cross-correlation: evaluating automated labeling
       of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
    """

    print('\nRestricted ANTs Registration')

     # filenames needed for intermediate results
    output_dir = _output_dir_4saving(output_dir, source_images[0])

    transformed_source_files = []
    for idx,source_image in enumerate(source_images):
        transformed_source_files.append(os.path.join(output_dir,
                                    _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='rarg-def'+str(idx))))

    mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='rarg-map'))

    inverse_mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='rarg-invmap'))
    if save_data:
        if overwrite is False \
            and os.path.isfile(mapping_file) \
            and os.path.isfile(inverse_mapping_file) :

            missing = False
            for trans_file in transformed_source_files:
                if not os.path.isfile(trans_file):
                    missing = True
            
            if not missing:
                print("skip computation (use existing results)")
                transformed = []
                for trans_file in transformed_source_files:
                    transformed.append(trans_file)
                output = {'transformed_sources': transformed,
                      'transformed_source': transformed[0],
                      'mapping': mapping_file,
                      'inverse': inverse_mapping_file}
                return output

        
    # build specific output names for the intermediate steps
    if file_name is not None:
        tmp = 'tmp_'+os.path.basename(file_name)
    else:
        tmp = 'tmp_'+os.path.basename(source_images[0])
 
 
    # find center (use non-zero values from both source and target)
    mask = (load_volume(source_images[0]).get_fdata()!=0)
    for img in source_images[1:]:
        mask = mask + (load_volume(img).get_fdata()!=0)
    for img in target_images[0:]:
        mask = mask + (load_volume(img).get_fdata()!=0)
    center = int(scipy.ndimage.center_of_mass(mask)[axis])
    end = mask.shape[axis]
    start = 0
    print('center: '+str(center))
    
    # from center register toward endpoints, then go back (?)
    
    # load input image to use as moving result
    sources = []
    for idx,src in enumerate(source_images):
        data = load_volume(src).get_fdata()
        sources.append(data)
        
    targets = []
    for idx,trg in enumerate(target_images):
        data = load_volume(trg).get_fdata()
        targets.append(data)
    
    mapping = numpy.zeros((targets[0].shape[0],targets[0].shape[1],targets[0].shape[2],3))
    inverse = numpy.zeros((sources[0].shape[0],sources[0].shape[1],sources[0].shape[2],3))
    
    if direction=='symmetric':
        # forward and backward from center
        for loc in range(center,end):
            print('aligning '+str(loc))     
            curr_depth = min(depth+1,loc-center+1)
            
            src_imgs = []
            trg_imgs = []
            img_weights = []
            # stack source sections, target sections, and previously registered sections
            for idx,src in enumerate(source_images):
                # current image (multiple times)
                source = sources[idx]
                src_img = tmp+'_src'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = source[loc,:,:]
                elif axis==1: 
                    data = source[:,loc,:]
                elif axis==2:
                    data = source[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(src_img,img)
                src_imgs.append(src_img)
                
                # target images
                target = targets[idx]
                trg_img = tmp+'_trg'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = target[loc,:,:]
                elif axis==1: 
                    data = target[:,loc,:]
                elif axis==2:
                    data = target[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(trg_img,img)
                trg_imgs.append(trg_img)
    
                img_weights.append(1.0)
                
                # mirror image (multiple times)
                mir = 2*center-loc
                source = sources[idx]
                src_img = tmp+'_src'+str(idx)+'_mirror.nii.gz'
                if axis==0: 
                    data = source[mir,:,:]
                elif axis==1: 
                    data = source[:,mir,:]
                elif axis==2:
                    data = source[:,:,mir]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(src_img,img)
                src_imgs.append(src_img)
                
                # target images
                target = targets[idx]
                trg_img = tmp+'_trg'+str(idx)+'_mirror.nii.gz'
                if axis==0: 
                    data = target[mir,:,:]
                elif axis==1: 
                    data = target[:,mir,:]
                elif axis==2:
                    data = target[:,:,mir]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(trg_img,img)
                trg_imgs.append(trg_img)
    
                img_weights.append(1.0)
                
                # previous images as additional target
                for dep in range(1, curr_depth):
                    trg_img = tmp+'_src'+str(idx)+'_depth'+str(dep)+'.nii.gz'
                    
                    if axis==0: 
                        data = source[loc-dep,:,:]
                    elif axis==1: 
                        data = source[:,loc-dep,:]
                    elif axis==2:
                        data = source[:,:,loc-dep]
                        
                    img = nibabel.Nifti1Image(data,None,None)
                    save_volume(trg_img,img)
                    trg_imgs.append(trg_img)
                    # add also current image as source
                    src_imgs.append(src_img)
            
                    img_weights.append(1.0/depth)
                    
                # previous images as additional target
                for dep in range(1, curr_depth):
                    trg_img = tmp+'_src'+str(idx)+'_mdepth'+str(dep)+'.nii.gz'
                    
                    if axis==0: 
                        data = source[mir+dep,:,:]
                    elif axis==1: 
                        data = source[:,mir+dep,:]
                    elif axis==2:
                        data = source[:,:,mir+dep]
                        
                    img = nibabel.Nifti1Image(data,None,None)
                    save_volume(trg_img,img)
                    trg_imgs.append(trg_img)
                    # add also current image as source
                    src_imgs.append(src_img)
            
                    img_weights.append(1.0/depth)
                    
                    
            step = embedded_antspy_2d_multi(src_imgs, trg_imgs,
                            image_weights=img_weights,
                            run_rigid=run_rigid,
                            rigid_iterations=rigid_iterations,
                            run_similarity=run_similarity,
                            similarity_iterations=similarity_iterations,
                            run_affine=run_affine,
                            affine_iterations=affine_iterations,
                            run_syn=run_syn,
                            coarse_iterations=coarse_iterations,
                            medium_iterations=medium_iterations, 
                            fine_iterations=fine_iterations,
                            scaling_factor=scaling_factor,
                            cost_function=cost_function,
                            interpolation=interpolation,
                            regularization=regularization,
                            convergence=convergence,
                            mask_zero=mask_zero,
                            ignore_affine=False, ignore_orient=False, ignore_res=False,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None)
                    
            # copy results into source image, mappings
            for idx,src in enumerate(source_images):
                if axis==0: 
                    sources[idx][loc,:,:] = step['transformed_sources'][idx*(2*curr_depth)+0].get_fdata()
                elif axis==1: 
                    sources[idx][:,loc,:] = step['transformed_sources'][idx*(2*curr_depth)+0].get_fdata()
                elif axis==2:
                    sources[idx][:,:,loc] = step['transformed_sources'][idx*(2*curr_depth)+0].get_fdata()
                if axis==0: 
                    sources[idx][mir,:,:] = step['transformed_sources'][idx*(2*curr_depth)+1].get_fdata()
                elif axis==1: 
                    sources[idx][:,mir,:] = step['transformed_sources'][idx*(2*curr_depth)+1].get_fdata()
                elif axis==2:
                    sources[idx][:,:,mir] = step['transformed_sources'][idx*(2*curr_depth)+1].get_fdata()
            
            if axis==0: 
                mapping[loc,:,:,0] = loc
                mapping[loc,:,:,1] = step['mapping'].get_fdata()[:,:,0]
                mapping[loc,:,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[loc,:,:,0] = loc
                inverse[loc,:,:,1] = step['inverse'].get_fdata()[:,:,0]
                inverse[loc,:,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==1: 
                mapping[:,loc,:,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,loc,:,1] = loc
                mapping[:,loc,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[:,loc,:,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,loc,:,1] = loc
                inverse[:,loc,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==2:
                mapping[:,:,loc,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,:,loc,1] = step['mapping'].get_fdata()[:,:,1]
                mapping[:,:,loc,2] = loc
                inverse[:,:,loc,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,:,loc,1] = step['inverse'].get_fdata()[:,:,1]
                inverse[:,:,loc,2] = loc
        
        for loc in range(end-1,center+1):
            print('aligning '+str(loc))     
            curr_depth = min(depth+1,end-1-loc+1)
            
            src_imgs = []
            trg_imgs = []
            img_weights = []
            # stack source sections, target sections, and previously registered sections
            for idx,src in enumerate(source_images):
                # current image (multiple times)
                source = sources[idx]
                src_img = tmp+'_src'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = source[loc,:,:]
                elif axis==1: 
                    data = source[:,loc,:]
                elif axis==2:
                    data = source[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(src_img,img)
                src_imgs.append(src_img)
                
                # target images
                target = targets[idx]
                trg_img = tmp+'_trg'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = target[loc,:,:]
                elif axis==1: 
                    data = target[:,loc,:]
                elif axis==2:
                    data = target[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(trg_img,img)
                trg_imgs.append(trg_img)
    
                img_weights.append(1.0)
                
                # mirror image (multiple times)
                mir = 2*center-loc
                source = sources[idx]
                src_img = tmp+'_src'+str(idx)+'_mirror.nii.gz'
                if axis==0: 
                    data = source[mir,:,:]
                elif axis==1: 
                    data = source[:,mir,:]
                elif axis==2:
                    data = source[:,:,mir]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(src_img,img)
                src_imgs.append(src_img)
                
                # target images
                target = targets[idx]
                trg_img = tmp+'_trg'+str(idx)+'_mirror.nii.gz'
                if axis==0: 
                    data = target[mir,:,:]
                elif axis==1: 
                    data = target[:,mir,:]
                elif axis==2:
                    data = target[:,:,mir]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(trg_img,img)
                trg_imgs.append(trg_img)
    
                img_weights.append(1.0)
                
                # previous images as additional target
                for dep in range(1, curr_depth):
                    trg_img = tmp+'_src'+str(idx)+'_depth'+str(dep)+'.nii.gz'
                    
                    if axis==0: 
                        data = source[loc+dep,:,:]
                    elif axis==1: 
                        data = source[:,loc+dep,:]
                    elif axis==2:
                        data = source[:,:,loc+dep]
                        
                    img = nibabel.Nifti1Image(data,None,None)
                    save_volume(trg_img,img)
                    trg_imgs.append(trg_img)
                    # add also current image as source
                    src_imgs.append(src_img)
            
                    img_weights.append(1.0/depth)
                    
                # previous images as additional target
                for dep in range(1, curr_depth):
                    trg_img = tmp+'_src'+str(idx)+'_mdepth'+str(dep)+'.nii.gz'
                    
                    if axis==0: 
                        data = source[mir-dep,:,:]
                    elif axis==1: 
                        data = source[:,mir-dep,:]
                    elif axis==2:
                        data = source[:,:,mir-dep]
                        
                    img = nibabel.Nifti1Image(data,None,None)
                    save_volume(trg_img,img)
                    trg_imgs.append(trg_img)
                    # add also current image as source
                    src_imgs.append(src_img)
            
                    img_weights.append(1.0/depth)
                    
                    
            step = embedded_antspy_2d_multi(src_imgs, trg_imgs,
                            image_weights=img_weights,
                            run_rigid=run_rigid,
                            rigid_iterations=rigid_iterations,
                            run_similarity=run_similarity,
                            similarity_iterations=similarity_iterations,
                            run_affine=run_affine,
                            affine_iterations=affine_iterations,
                            run_syn=run_syn,
                            coarse_iterations=coarse_iterations,
                            medium_iterations=medium_iterations, 
                            fine_iterations=fine_iterations,
                            scaling_factor=scaling_factor,
                            cost_function=cost_function,
                            interpolation=interpolation,
                            regularization=regularization,
                            convergence=convergence,
                            mask_zero=mask_zero,
                            ignore_affine=False, ignore_orient=False, ignore_res=False,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None)
                    
            # copy results into source image, mappings
            for idx,src in enumerate(source_images):
                if axis==0: 
                    sources[idx][loc,:,:] = step['transformed_sources'][idx*(2*curr_depth)+0].get_fdata()
                elif axis==1: 
                    sources[idx][:,loc,:] = step['transformed_sources'][idx*(2*curr_depth)+0].get_fdata()
                elif axis==2:
                    sources[idx][:,:,loc] = step['transformed_sources'][idx*(2*curr_depth)+0].get_fdata()
                if axis==0: 
                    sources[idx][mir,:,:] = step['transformed_sources'][idx*(2*curr_depth)+1].get_fdata()
                elif axis==1: 
                    sources[idx][:,mir,:] = step['transformed_sources'][idx*(2*curr_depth)+1].get_fdata()
                elif axis==2:
                    sources[idx][:,:,mir] = step['transformed_sources'][idx*(2*curr_depth)+1].get_fdata()
            
            if axis==0: 
                mapping[loc,:,:,0] = loc
                mapping[loc,:,:,1] = step['mapping'].get_fdata()[:,:,0]
                mapping[loc,:,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[loc,:,:,0] = loc
                inverse[loc,:,:,1] = step['inverse'].get_fdata()[:,:,0]
                inverse[loc,:,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==1: 
                mapping[:,loc,:,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,loc,:,1] = loc
                mapping[:,loc,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[:,loc,:,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,loc,:,1] = loc
                inverse[:,loc,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==2:
                mapping[:,:,loc,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,:,loc,1] = step['mapping'].get_fdata()[:,:,1]
                mapping[:,:,loc,2] = loc
                inverse[:,:,loc,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,:,loc,1] = step['inverse'].get_fdata()[:,:,1]
                inverse[:,:,loc,2] = loc
        
    elif direction=='center':
    
        # forward from center
        for loc in range(center,end):
            print('aligning '+str(loc))  
            curr_depth = min(depth+1,loc-center+1)
            
            src_imgs = []
            trg_imgs = []
            img_weights = []
            # stack source sections, target sections, and previously registered sections
            for idx,src in enumerate(source_images):
                # current image (multiple times)
                source = sources[idx]
                src_img = tmp+'_src'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = source[loc,:,:]
                elif axis==1: 
                    data = source[:,loc,:]
                elif axis==2:
                    data = source[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(src_img,img)
                src_imgs.append(src_img)
                
                # target images
                target = targets[idx]
                trg_img = tmp+'_trg'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = target[loc,:,:]
                elif axis==1: 
                    data = target[:,loc,:]
                elif axis==2:
                    data = target[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(trg_img,img)
                trg_imgs.append(trg_img)
    
                img_weights.append(1.0)
                
                # previous images as additional target
                for dep in range(1, curr_depth):
                    trg_img = tmp+'_src'+str(idx)+'_depth'+str(dep)+'.nii.gz'
                    
                    if axis==0: 
                        data = source[loc-dep,:,:]
                    elif axis==1: 
                        data = source[:,loc-dep,:]
                    elif axis==2:
                        data = source[:,:,loc-dep]
                        
                    img = nibabel.Nifti1Image(data,None,None)
                    save_volume(trg_img,img)
                    trg_imgs.append(trg_img)
                    # add also current image as source
                    src_imgs.append(src_img)
            
                    img_weights.append(1.0/depth)
                    
                    
            step = embedded_antspy_2d_multi(src_imgs, trg_imgs,
                            image_weights=img_weights,
                            run_rigid=run_rigid,
                            rigid_iterations=rigid_iterations,
                            run_similarity=run_similarity,
                            similarity_iterations=similarity_iterations,
                            run_affine=run_affine,
                            affine_iterations=affine_iterations,
                            run_syn=run_syn,
                            coarse_iterations=coarse_iterations,
                            medium_iterations=medium_iterations, 
                            fine_iterations=fine_iterations,
                            scaling_factor=scaling_factor,
                            cost_function=cost_function,
                            interpolation=interpolation,
                            regularization=regularization,
                            convergence=convergence,
                            mask_zero=mask_zero,
                            ignore_affine=False, ignore_orient=False, ignore_res=False,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None)
                    
            # copy results into source image, mappings
            for idx,src in enumerate(source_images):
                if axis==0: 
                    sources[idx][loc,:,:] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
                elif axis==1: 
                    sources[idx][:,loc,:] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
                elif axis==2:
                    sources[idx][:,:,loc] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
            
            if axis==0: 
                mapping[loc,:,:,0] = loc
                mapping[loc,:,:,1] = step['mapping'].get_fdata()[:,:,0]
                mapping[loc,:,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[loc,:,:,0] = loc
                inverse[loc,:,:,1] = step['inverse'].get_fdata()[:,:,0]
                inverse[loc,:,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==1: 
                mapping[:,loc,:,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,loc,:,1] = loc
                mapping[:,loc,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[:,loc,:,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,loc,:,1] = loc
                inverse[:,loc,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==2:
                mapping[:,:,loc,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,:,loc,1] = step['mapping'].get_fdata()[:,:,1]
                mapping[:,:,loc,2] = loc
                inverse[:,:,loc,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,:,loc,1] = step['inverse'].get_fdata()[:,:,1]
                inverse[:,:,loc,2] = loc
        
        
        # backward from center
        for loc in range(center-1,-1,-1):
            print('aligning '+str(loc))       
            curr_depth = min(depth+1,center-loc+1)
            
            src_imgs = []
            trg_imgs = []
            img_weights = []
            # stack source sections, target sections, and previously registered sections
            for idx,src in enumerate(source_images):
                # current image (multiple times)
                source = sources[idx]
                src_img = tmp+'_src'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = source[loc,:,:]
                elif axis==1: 
                    data = source[:,loc,:]
                elif axis==2:
                    data = source[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(src_img,img)
                src_imgs.append(src_img)
                
                # target images
                target = targets[idx]
                trg_img = tmp+'_trg'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = target[loc,:,:]
                elif axis==1: 
                    data = target[:,loc,:]
                elif axis==2:
                    data = target[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(trg_img,img)
                trg_imgs.append(trg_img)
    
                img_weights.append(1.0)
                
                # previous images as additional target
                for dep in range(1, curr_depth):
                    trg_img = tmp+'_src'+str(idx)+'_depth'+str(dep)+'.nii.gz'
                    
                    if axis==0: 
                        data = source[loc+dep,:,:]
                    elif axis==1: 
                        data = source[:,loc+dep,:]
                    elif axis==2:
                        data = source[:,:,loc+dep]
                        
                    img = nibabel.Nifti1Image(data,None,None)
                    save_volume(trg_img,img)
                    trg_imgs.append(trg_img)
                    # add also current image as source
                    src_imgs.append(src_img)
            
                    img_weights.append(1.0/depth)
                    
                    
            step = embedded_antspy_2d_multi(src_imgs, trg_imgs,
                            image_weights=img_weights,
                            run_rigid=run_rigid,
                            rigid_iterations=rigid_iterations,
                            run_similarity=run_similarity,
                            similarity_iterations=similarity_iterations,
                            run_affine=run_affine,
                            affine_iterations=affine_iterations,
                            run_syn=run_syn,
                            coarse_iterations=coarse_iterations,
                            medium_iterations=medium_iterations, 
                            fine_iterations=fine_iterations,
                            scaling_factor=scaling_factor,
                            cost_function=cost_function,
                            interpolation=interpolation,
                            regularization=regularization,
                            convergence=convergence,
                            mask_zero=mask_zero,
                            ignore_affine=False, ignore_orient=False, ignore_res=False,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None)
            
            # copy results into source image, mappings
            for idx,src in enumerate(source_images):
                if axis==0: 
                    sources[idx][loc,:,:] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
                elif axis==1: 
                    sources[idx][:,loc,:] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
                elif axis==2:
                    sources[idx][:,:,loc] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
                    
            if axis==0: 
                mapping[loc,:,:,0] = loc
                mapping[loc,:,:,1] = step['mapping'].get_fdata()[:,:,0]
                mapping[loc,:,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[loc,:,:,0] = loc
                inverse[loc,:,:,1] = step['inverse'].get_fdata()[:,:,0]
                inverse[loc,:,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==1: 
                mapping[:,loc,:,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,loc,:,1] = loc
                mapping[:,loc,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[:,loc,:,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,loc,:,1] = loc
                inverse[:,loc,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==2:
                mapping[:,:,loc,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,:,loc,1] = step['mapping'].get_fdata()[:,:,1]
                mapping[:,:,loc,2] = loc
                inverse[:,:,loc,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,:,loc,1] = step['inverse'].get_fdata()[:,:,1]
                inverse[:,:,loc,2] = loc
                    
    elif direction=='sweep':
    
        # forward from start
        for loc in range(start,end):
            print('aligning '+str(loc))  
            curr_depth = min(depth+1,loc-start+1)
            
            src_imgs = []
            trg_imgs = []
            img_weights = []
            # stack source sections, target sections, and previously registered sections
            for idx,src in enumerate(source_images):
                # current image (multiple times)
                source = sources[idx]
                src_img = tmp+'_src'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = source[loc,:,:]
                elif axis==1: 
                    data = source[:,loc,:]
                elif axis==2:
                    data = source[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(src_img,img)
                src_imgs.append(src_img)
                
                # target images
                target = targets[idx]
                trg_img = tmp+'_trg'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = target[loc,:,:]
                elif axis==1: 
                    data = target[:,loc,:]
                elif axis==2:
                    data = target[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(trg_img,img)
                trg_imgs.append(trg_img)
    
                img_weights.append(1.0)
                
                # previous images as additional target
                for dep in range(1, curr_depth):
                    trg_img = tmp+'_src'+str(idx)+'_depth'+str(dep)+'.nii.gz'
                    
                    if axis==0: 
                        data = source[loc-dep,:,:]
                    elif axis==1: 
                        data = source[:,loc-dep,:]
                    elif axis==2:
                        data = source[:,:,loc-dep]
                        
                    img = nibabel.Nifti1Image(data,None,None)
                    save_volume(trg_img,img)
                    trg_imgs.append(trg_img)
                    # add also current image as source
                    src_imgs.append(src_img)
            
                    img_weights.append(1.0/depth)
                    
                    
            step = embedded_antspy_2d_multi(src_imgs, trg_imgs,
                            image_weights=img_weights,
                            run_rigid=run_rigid,
                            rigid_iterations=rigid_iterations,
                            run_similarity=run_similarity,
                            similarity_iterations=similarity_iterations,
                            run_affine=run_affine,
                            affine_iterations=affine_iterations,
                            run_syn=run_syn,
                            coarse_iterations=coarse_iterations,
                            medium_iterations=medium_iterations, 
                            fine_iterations=fine_iterations,
                            scaling_factor=scaling_factor,
                            cost_function=cost_function,
                            interpolation=interpolation,
                            regularization=regularization,
                            convergence=convergence,
                            mask_zero=mask_zero,
                            ignore_affine=False, ignore_orient=False, ignore_res=False,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None)
                    
            # copy results into source image, mappings
            for idx,src in enumerate(source_images):
                if axis==0: 
                    sources[idx][loc,:,:] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
                elif axis==1: 
                    sources[idx][:,loc,:] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
                elif axis==2:
                    sources[idx][:,:,loc] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
            
            if axis==0: 
                mapping[loc,:,:,0] = loc
                mapping[loc,:,:,1] = step['mapping'].get_fdata()[:,:,0]
                mapping[loc,:,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[loc,:,:,0] = loc
                inverse[loc,:,:,1] = step['inverse'].get_fdata()[:,:,0]
                inverse[loc,:,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==1: 
                mapping[:,loc,:,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,loc,:,1] = loc
                mapping[:,loc,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[:,loc,:,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,loc,:,1] = loc
                inverse[:,loc,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==2:
                mapping[:,:,loc,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,:,loc,1] = step['mapping'].get_fdata()[:,:,1]
                mapping[:,:,loc,2] = loc
                inverse[:,:,loc,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,:,loc,1] = step['inverse'].get_fdata()[:,:,1]
                inverse[:,:,loc,2] = loc
        
        
        # backward from end
        for loc in range(end-1,-1,-1):
            print('aligning '+str(loc))       
            curr_depth = min(depth+1,end-1-loc+1)
            
            src_imgs = []
            trg_imgs = []
            img_weights = []
            # stack source sections, target sections, and previously registered sections
            for idx,src in enumerate(source_images):
                # current image (multiple times)
                source = sources[idx]
                src_img = tmp+'_src'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = source[loc,:,:]
                elif axis==1: 
                    data = source[:,loc,:]
                elif axis==2:
                    data = source[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(src_img,img)
                src_imgs.append(src_img)
                
                # target images
                target = targets[idx]
                trg_img = tmp+'_trg'+str(idx)+'_current.nii.gz'
                if axis==0: 
                    data = target[loc,:,:]
                elif axis==1: 
                    data = target[:,loc,:]
                elif axis==2:
                    data = target[:,:,loc]
                    
                img = nibabel.Nifti1Image(data,None,None)
                save_volume(trg_img,img)
                trg_imgs.append(trg_img)
    
                img_weights.append(1.0)
                
                # previous images as additional target
                for dep in range(1, curr_depth):
                    trg_img = tmp+'_src'+str(idx)+'_depth'+str(dep)+'.nii.gz'
                    
                    if axis==0: 
                        data = source[loc+dep,:,:]
                    elif axis==1: 
                        data = source[:,loc+dep,:]
                    elif axis==2:
                        data = source[:,:,loc+dep]
                        
                    img = nibabel.Nifti1Image(data,None,None)
                    save_volume(trg_img,img)
                    trg_imgs.append(trg_img)
                    # add also current image as source
                    src_imgs.append(src_img)
            
                    img_weights.append(1.0/depth)
                    
                    
            step = embedded_antspy_2d_multi(src_imgs, trg_imgs,
                            image_weights=img_weights,
                            run_rigid=run_rigid,
                            rigid_iterations=rigid_iterations,
                            run_similarity=run_similarity,
                            similarity_iterations=similarity_iterations,
                            run_affine=run_affine,
                            affine_iterations=affine_iterations,
                            run_syn=run_syn,
                            coarse_iterations=coarse_iterations,
                            medium_iterations=medium_iterations, 
                            fine_iterations=fine_iterations,
                            scaling_factor=scaling_factor,
                            cost_function=cost_function,
                            interpolation=interpolation,
                            regularization=regularization,
                            convergence=convergence,
                            mask_zero=mask_zero,
                            ignore_affine=False, ignore_orient=False, ignore_res=False,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None)
            
            # copy results into source image, mappings
            for idx,src in enumerate(source_images):
                if axis==0: 
                    sources[idx][loc,:,:] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
                elif axis==1: 
                    sources[idx][:,loc,:] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
                elif axis==2:
                    sources[idx][:,:,loc] = step['transformed_sources'][idx*(curr_depth)].get_fdata()
                    
            if axis==0: 
                mapping[loc,:,:,0] = loc
                mapping[loc,:,:,1] = step['mapping'].get_fdata()[:,:,0]
                mapping[loc,:,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[loc,:,:,0] = loc
                inverse[loc,:,:,1] = step['inverse'].get_fdata()[:,:,0]
                inverse[loc,:,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==1: 
                mapping[:,loc,:,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,loc,:,1] = loc
                mapping[:,loc,:,2] = step['mapping'].get_fdata()[:,:,1]
                inverse[:,loc,:,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,loc,:,1] = loc
                inverse[:,loc,:,2] = step['inverse'].get_fdata()[:,:,1]
            elif axis==2:
                mapping[:,:,loc,0] = step['mapping'].get_fdata()[:,:,0]
                mapping[:,:,loc,1] = step['mapping'].get_fdata()[:,:,1]
                mapping[:,:,loc,2] = loc
                inverse[:,:,loc,0] = step['inverse'].get_fdata()[:,:,0]
                inverse[:,:,loc,1] = step['inverse'].get_fdata()[:,:,1]
                inverse[:,:,loc,2] = loc
                    
	    
    # clean up intermediate files
    for idx,src in enumerate(source_images):
       src_img = tmp+'_src'+str(idx)+'_current.nii.gz'
       if os.path.exists(src_img): os.remove(src_img)
       trg_img = tmp+'_trg'+str(idx)+'_current.nii.gz'
       if os.path.exists(trg_img): os.remove(trg_img)
       for dep in range(1, depth+1):
           trg_img = tmp+'_src'+str(idx)+'_depth'+str(dep)+'.nii.gz'
           if os.path.exists(trg_img): os.remove(trg_img)
             
    # build final results
    transformed = []
    img = load_volume(target_images[0])
    for idx,trg in enumerate(target_images):
        transformed.append(nibabel.Nifti1Image(sources[idx],img.affine,img.header))
    mapping = nibabel.Nifti1Image(mapping,img.affine,img.header)
    inverse = nibabel.Nifti1Image(inverse,img.affine,img.header)
        
    if save_data:
        for idx,trans in enumerate(transformed):
            save_volume(transformed_source_files[idx],trans)
        save_volume(mapping_file,mapping)
        save_volume(inverse_mapping_file,inverse)
        
        transformed = transformed_source_files
        mapping = mapping_file
        inverse = inverse_mapping_file
        
    output = {'transformed_sources': transformed,
              'transformed_source': transformed[0],
              'mapping': mapping,
              'inverse': inverse}

    return output
