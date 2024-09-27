# basic dependencies
import os
import sys
from glob import glob
import math

# main dependencies: numpy, nibabel, ants
import numpy
import nibabel
import ants.utils

# nighresjava and nighres functions
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir
from ..surface import probability_to_levelset
from . import embedded_antspy_multi, apply_coordinate_mappings

# convenience labels
X=0
Y=1
Z=2
T=3

def focused_antspy(source_images, target_images, source_label=None, target_label=None,
                    label_list=None, label_distance=3.0,
                    run_rigid=True,
                    rigid_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
					scaling_factor=8,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False, smooth_mask=0.0,
					ignore_affine=False, ignore_header=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Focused ANTSpy Registration

    Runs the rigid and/or Symmetric Normalization (SyN) algorithm of ANTs in a two-step
    approach to fine-tune coregistration of specific regions inside images. The first step
    aligns the images globally, then the second step masks out regions far from the labels
    of interest.

    Parameters
    ----------
    source_images: [niimg]
        Image list to register
    target_images: [niimg]
        Reference image list to match
    source_label: [niimg]
        Labeling of the source images (optional, but at least one labeling is needed)
    target_label: [niimg]
        Labeling of the target images (optional, but at least one labeling is needed)
    label_list: [int]
        List of labels to be included in the region of interest (if not given, labels>0 are used)
    label_distance: float
        Maximum distance around the region of interest to keep as a ratio of the region internal
        thickness (default is 3.0)
    run_rigid: bool
        Whether or not to run a rigid registration first (default is False)
    rigid_iterations: float
        Number of iterations in the rigid step (default is 1000)
    run_affine: bool
        Whether or not to run a affine registration first (default is False)
    affine_iterations: float
        Number of iterations in the affine step (default is 1000)
    run_syn: bool
        Whether or not to run a SyN registration (default is True)
    coarse_iterations: float
        Number of iterations at the coarse level (default is 40)
    medium_iterations: float
        Number of iterations at the medium level (default is 50)
    fine_iterations: float
        Number of iterations at the fine level (default is 40)
    cost_function: {'CrossCorrelation', 'MutualInformation'}
        Cost function for the registration (default is 'MutualInformation')
    interpolation: {'NearestNeighbor', 'Linear'}
        Interpolation for the registration result (default is 'NearestNeighbor')
    regularization: {'Low', 'Medium', 'High'}
        Regularization preset for the SyN deformation (default is 'Medium')
    convergence: float
        Threshold for convergence, can make the algorithm very slow (default is convergence)
    mask_zero: bool
        Mask regions with zero value using ANTs masking option (default is False)
    smooth_mask: float
        Smoothly mask regions within a given ratio of the object's thickness,
        in [0.0, 1.0] (default is 0.0). This does not use ANTs masking.
    ignore_affine: bool
        Ignore the affine matrix information extracted from the image header
        (default is False)
    ignore_header: bool
        Ignore the orientation information and affine matrix information
        extracted from the image header (default is False)
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

        * transformed_sources ([niimg]): Deformed focused source image list (_ants_def0,1,...)
        * focused_targets ([niimg]): Focused target image list (_ants_def0,1,...)
        * mapping (niimg): Coordinate mapping from source to target (_ants_map)
        * inverse (niimg): Inverse coordinate mapping from target to source (_ants_invmap)

    Notes
    ----------
    he SyN algorithm is part of the ANTs software by Brian Avants and colleagues [1]_. 
    Parameters have been set to values commonly found in neuroimaging scripts online, but not 
    necessarily optimal.

    References
    ----------
    .. [1] Avants et al (2008), Symmetric diffeomorphic
       image registration with cross-correlation: evaluating automated labeling
       of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
    """

    print('\nFocused ANTs Registration')

     # filenames needed for intermediate results
    output_dir = _output_dir_4saving(output_dir, source_images[0])

    transformed_source_files = []
    for idx,source_image in enumerate(source_images):
        transformed_source_files.append(os.path.join(output_dir,
                                    _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='freg-def'+str(idx))))

    focused_target_files = []
    for idx,target_image in enumerate(target_images):
        focused_target_files.append(os.path.join(output_dir,
                                    _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=target_image,
                                   suffix='freg-trg'+str(idx))))

    mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='freg-map'))

    inverse_mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='freg-invmap'))
    if save_data:
        if overwrite is False \
            and os.path.isfile(mapping_file) \
            and os.path.isfile(inverse_mapping_file) :

            missing = False
            for trans_file in transformed_source_files:
                if not os.path.isfile(trans_file):
                    missing = True
            for focus_file in focused_target_files:
                if not os.path.isfile(focus_file):
                    missing = True

            if not missing:
                print("skip computation (use existing results)")
                transformed = []
                for trans_file in transformed_source_files:
                    transformed.append(trans_file)
                focused = []
                for focus_file in focused_target_files:
                    focused.append(focus_file)
                output = {'transformed_sources': transformed,
                      'transformed_source': transformed[0],
                      'focused_targets': focused,
                      'focused_target': focused[0],
                      'mapping': mapping_file,
                      'inverse': inverse_mapping_file}
                return output


    print('\nStep 1: Global Registration')
    step1 = embedded_antspy_multi(source_images, target_images,
                    run_rigid, rigid_iterations, run_affine, affine_iterations,
                    run_syn, coarse_iterations, medium_iterations, fine_iterations,
					scaling_factor, cost_function, interpolation, regularization, 
					convergence, mask_zero, smooth_mask, ignore_affine, ignore_header,
					save_data, overwrite, output_dir, file_name=None) 
	
    print('\nStep 2: Focused Registration')
    
    print('\nBuild focused images')
    
    if target_label is not None:
        print('\ntarget focus region')
        img = load_volume(source_label)
        data = img.get_fdata()
        
        mask = numpy.zeros(data.shape)
        if label_list is not None:
            for lb in label_list:
                mask = mask + 1.0*(data==lb)
        else:
            mask = 1.0*(data>0)
            
        img = nibabel.Nifti1Image(mask,img.affine,img.header)    
    
        focus = probability_to_levelset(img)['result'].get_fdata()
        maxfocus = -2.0*numpy.min(focus)
        print('\nthickness: '+str(maxfocus))
        
        mask = 1.0 - focus/(label_distance*maxfocus)
        mask[mask>1] = 1.0
        mask[mask<0] = 0.0
        
        for idx,target in enumerate(target_images):
            img = load_volume(target)
            img = nibabel.Nifti1Image(mask*img.get_fdata(),img.affine,img.header)
            save_volume(focused_target_files[idx],img)
        
        if source_label is None:
            for idx,transformed in enumerate(step1['transformed_sources']):
                img = load_volume(transformed)
                img = nibabel.Nifti1Image(mask*img.get_fdata(),img.affine,img.header)
                save_volume(transformed_source_files[idx],img)
            
    if source_label is not None:
        print('\nsource focus region')
        label_img = apply_coordinate_mappings(source_label, mapping1=step1['mapping'],
                        interpolation="nearest", padding="zero",
                        zero_border=0, check_boundaries=False,
                        save_data=False)['result']
        img = load_volume(label_img)
        data = img.get_fdata()
        
        mask = numpy.zeros(data.shape)
        if label_list is not None:
            for lb in label_list:
                mask = mask + 1.0*(data==lb)
        else:
            mask = 1.0*(data>0)
            
        img = nibabel.Nifti1Image(mask,img.affine,img.header)    
    
        focus = probability_to_levelset(img)['result'].get_fdata()
        maxfocus = -2.0*numpy.min(focus)
        print('\nthickness: '+str(maxfocus))
        
        mask = 1.0 - focus/(label_distance*maxfocus)
        mask[mask>1] = 1.0
        mask[mask<0] = 0.0
        
        for idx,transformed in enumerate(step1['transformed_sources']):
            img = load_volume(transformed)
            img = nibabel.Nifti1Image(mask*img.get_fdata(),img.affine,img.header)
            save_volume(transformed_source_files[idx],img)
        
        if target_label is None:
            for idx,target in enumerate(target_images):
                img = load_volume(target)
                img = nibabel.Nifti1Image(mask*img.get_fdata(),img.affine,img.header)
                save_volume(focused_target_files[idx],img)        
    
    print('\nRegister focused images')
    step2 = embedded_antspy_multi(transformed_source_files, focused_target_files,
                    run_rigid=False,run_affine=False,run_syn=True,
                    coarse_iterations=coarse_iterations,
                    medium_iterations=medium_iterations, 
                    fine_iterations=fine_iterations,
					scaling_factor=scaling_factor,
					cost_function=cost_function,
					interpolation=interpolation,
					regularization=regularization,
					convergence=convergence,
					mask_zero=False, smooth_mask=0.0, 
					ignore_affine=False, ignore_header=False,
					save_data=save_data, overwrite=overwrite, output_dir=output_dir, file_name=None) 

    print('\nCombine transformations')
    mapping = apply_coordinate_mappings(step1['mapping'], mapping1=step2['mapping'],
                        interpolation="linear", padding="closest",
                        zero_border=0, check_boundaries=False,
                        save_data=False)['result']
    save_volume(mapping_file, mapping)
    
    inverse = apply_coordinate_mappings(step2['inverse'], mapping1=step1['inverse'],
                        interpolation="linear", padding="closest",
                        zero_border=0, check_boundaries=False,
                        save_data=False)['result']
    save_volume(inverse_mapping_file, inverse)
    
    if not save_data:
        # rename final results
        for idx,trans_file in enumerate(step2['transformed_sources']):
            os.rename(trans_file, transformed_source_files[idx])

        # clean up intermediate files
        for img in step1['transformed_sources']:
            if os.path.exists(img): os.remove(img)
        if os.path.exists(step1['mapping']): os.remove(step1['mapping'])
        if os.path.exists(step1['inverse']): os.remove(step1['inverse'])

        for img in step2['transformed_sources']:
            if os.path.exists(img): os.remove(img)
        if os.path.exists(step2['mapping']): os.remove(step2['mapping'])
        if os.path.exists(step2['inverse']): os.remove(step2['inverse'])

        # collect saved outputs
        transformed = []
        for trans_file in transformed_source_files:
            transformed.append(load_volume(trans_file))
        focused = []
        for focus_file in focused_target_files:
            focused.append(load_volume(focus_file))
        output = {'transformed_sources': transformed,
              'transformed_source': transformed[0],
              'focused_targets': focused,
              'focused_target': focused[0],
              'mapping': load_volume(mapping_file),
              'inverse': load_volume(inverse_mapping_file)}

        # remove output files if *not* saved
        for trans_image in transformed_source_files:
            if os.path.exists(trans_image): os.remove(trans_image)
        for focus_file in focused_target_files:
            if os.path.exists(focus_image): os.remove(focus_image)
        if os.path.exists(mapping_file): os.remove(mapping_file)
        if os.path.exists(inverse_mapping_file): os.remove(inverse_mapping_file)

        return output
    else:
        # rename final results
        for idx,trans_file in enumerate(step2['transformed_sources']):
            os.rename(trans_file, transformed_source_files[idx])

        # clean up intermediate files
        for img in step1['transformed_sources']:
            if os.path.exists(img): os.remove(img)
        if os.path.exists(step1['mapping']): os.remove(step1['mapping'])
        if os.path.exists(step1['inverse']): os.remove(step1['inverse'])

        for img in step2['transformed_sources']:
            if os.path.exists(img): os.remove(img)
        if os.path.exists(step2['mapping']): os.remove(step2['mapping'])
        if os.path.exists(step2['inverse']): os.remove(step2['inverse'])

        # collect saved outputs
        output = {'transformed_sources': transformed_source_files,
              'transformed_source': transformed_source_files[0],
              'focused_targets': focused_target_files,
              'focused_target': focused_target_files[0],
              'mapping': mapping_file,
              'inverse': inverse_mapping_file}

        return output
    