# basic dependencies
import os
import sys
import subprocess
from glob import glob
import math

# main dependencies: numpy, nibabel
import numpy
import nibabel

# nighresjava and nighres functions
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir

# convenience labels
X=0
Y=1
Z=2
T=3

def surface_antsreg(source_surface, target_surface,
                    max_dist=10.0,
                    run_rigid=True,
                    rigid_iterations=1000,
                    run_affine=True,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=100,
                    medium_iterations=70, fine_iterations=20,
					cost_function='Demons',
					interpolation='Linear',
					regularization='Low',
					convergence=1e-6,
					mask_zero=False,
					crop=True,
					ignore_affine=False, ignore_header=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Embedded ANTS Registration for surfaces

    Runs the rigid and/or Symmetric Normalization (SyN) algorithm of ANTs and
    formats the output deformations into voxel coordinate mappings as used in
    CBSTools registration and transformation routines. Uses all input contrasts
    with equal weights.

    Parameters
    ----------
    source_surface: niimg
        Levelset surface image to register
    target_surface: niimg
        Reference levelset surface image to match
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
    cost_function: {'LeastSquares', 'Demons'}
        Cost function for the registration (default is 'Demons')
    interpolation: {'NearestNeighbor', 'Linear'}
        Interpolation for the registration result (default is 'Linear')
    regularization: {'Low', 'Medium', 'High'}
        Regularization preset for the SyN deformation (default is 'Medium')
    convergence: float
        Threshold for convergence, can make the algorithm very slow (default is convergence)
    mask_zero: bool
        Mask regions with zero value
        (default is False)
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

        * mapping (niimg): Coordinate mapping from source to target (_ants_map)
        * inverse (niimg): Inverse coordinate mapping from target to source (_ants_invmap)

    Notes
    ----------
    Port of the CBSTools Java module by Pierre-Louis Bazin. The main algorithm
    is part of the ANTs software by Brian Avants and colleagues [1]_. Parameters
    have been set to values commonly found in neuroimaging scripts online, but
    not necessarily optimal.

    References
    ----------
    .. [1] Avants et al (2008), Symmetric diffeomorphic
       image registration with cross-correlation: evaluating automated labeling
       of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
    """

    print('\nEmbedded ANTs Registration Surfaces')
    # check if ants is installed to raise sensible error
    try:
        subprocess.run('antsRegistration', stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        sys.exit("\nCould not find command 'antsRegistration'. Make sure ANTs is"
                 " installed and can be accessed from the command line.")
    try:
        subprocess.run('antsApplyTransforms', stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        sys.exit("\nCould not find command 'antsApplyTransforms'. Make sure ANTs"
                 " is installed and can be accessed from the command line.")

    # make sure that saving related parameters are correct

     # output files needed for intermediate results
    output_dir = _output_dir_4saving(output_dir, source_surface)

    mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_surface,
                               suffix='ants-map'))

    inverse_mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_surface,
                               suffix='ants-invmap'))
    if save_data:
        if overwrite is False \
            and os.path.isfile(mapping_file) \
            and os.path.isfile(inverse_mapping_file) :

                print("skip computation (use existing results)")
                output = {'mapping': mapping_file,
                          'inverse': inverse_mapping_file}
                return output

    # cropping and masking do not work well together?
    if crop: mask_zero=False

    # load and get dimensions and resolution from input images
    source = load_volume(source_surface)
    # flip the data around, threshold
    source_ls = numpy.minimum(numpy.maximum(max_dist - source.get_data(),0.0),2.0*max_dist)
    if crop:
        # crop images for speed?
        src_xmin, src_xmax = numpy.where(numpy.any(source_ls>0.1, axis=(1,2)))[0][[0, -1]]
        src_ymin, src_ymax = numpy.where(numpy.any(source_ls>0.1, axis=(0,2)))[0][[0, -1]]
        src_zmin, src_zmax = numpy.where(numpy.any(source_ls>0.1, axis=(0,1)))[0][[0, -1]]
        
        source_ls = source_ls[src_xmin:src_xmax+1, src_ymin:src_ymax+1, src_zmin:src_zmax+1]
        
    src_img = nibabel.Nifti1Image(source_ls, source.affine, source.header)
    src_img.update_header()
    src_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                    rootfile=source_surface,
                                                    suffix='tmp_srcimg'))
    save_volume(src_img_file, src_img)
    source = load_volume(src_img_file)

    src_affine = source.affine
    src_header = source.header
    nsx = source.header.get_data_shape()[X]
    nsy = source.header.get_data_shape()[Y]
    nsz = source.header.get_data_shape()[Z]
    rsx = source.header.get_zooms()[X]
    rsy = source.header.get_zooms()[Y]
    rsz = source.header.get_zooms()[Z]

    orig_src_aff = source.affine
    orig_src_hdr = source.header


    target = load_volume(target_surface)
    # flip the data around
    target_ls = numpy.minimum(numpy.maximum(max_dist - target.get_data(),0.0),2.0*max_dist)
    if crop:
        # crop images for speed?
        trg_xmin, trg_xmax = numpy.where(numpy.any(target_ls>0.1, axis=(1,2)))[0][[0, -1]]
        trg_ymin, trg_ymax = numpy.where(numpy.any(target_ls>0.1, axis=(0,2)))[0][[0, -1]]
        trg_zmin, trg_zmax = numpy.where(numpy.any(target_ls>0.1, axis=(0,1)))[0][[0, -1]]
        
        target_ls = target_ls[trg_xmin:trg_xmax+1, trg_ymin:trg_ymax+1, trg_zmin:trg_zmax+1]
        
    trg_img = nibabel.Nifti1Image(target_ls, target.affine, target.header)
    trg_img.update_header()
    trg_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                    rootfile=target_surface,
                                                    suffix='tmp_trgimg'))
    save_volume(trg_img_file, trg_img)
    target = load_volume(trg_img_file)

    trg_affine = target.affine
    trg_header = target.header
    ntx = target.header.get_data_shape()[X]
    nty = target.header.get_data_shape()[Y]
    ntz = target.header.get_data_shape()[Z]
    rtx = target.header.get_zooms()[X]
    rty = target.header.get_zooms()[Y]
    rtz = target.header.get_zooms()[Z]

    orig_trg_aff = target.affine
    orig_trg_hdr = target.header

    # in case the affine transformations are not to be trusted: make them equal
    if ignore_affine or ignore_header:
        # create generic affine aligned with the orientation for the source
        new_affine = numpy.zeros((4,4))
        if ignore_header:
            new_affine[0][0] = rsx
            new_affine[1][1] = rsy
            new_affine[2][2] = rsz
            new_affine[0][3] = -rsx*nsx/2.0
            new_affine[1][3] = -rsy*nsy/2.0
            new_affine[2][3] = -rsz*nsz/2.0
        else:
            mx = numpy.argmax(numpy.abs([src_affine[0][0],src_affine[1][0],src_affine[2][0]]))
            my = numpy.argmax(numpy.abs([src_affine[0][1],src_affine[1][1],src_affine[2][1]]))
            mz = numpy.argmax(numpy.abs([src_affine[0][2],src_affine[1][2],src_affine[2][2]]))
            new_affine[mx][0] = rsx*numpy.sign(src_affine[mx][0])
            new_affine[my][1] = rsy*numpy.sign(src_affine[my][1])
            new_affine[mz][2] = rsz*numpy.sign(src_affine[mz][2])
            if (numpy.sign(src_affine[mx][0])<0):
                new_affine[mx][3] = rsx*nsx/2.0
            else:
                new_affine[mx][3] = -rsx*nsx/2.0

            if (numpy.sign(src_affine[my][1])<0):
                new_affine[my][3] = rsy*nsy/2.0
            else:
                new_affine[my][3] = -rsy*nsy/2.0

            if (numpy.sign(src_affine[mz][2])<0):
                new_affine[mz][3] = rsz*nsz/2.0
            else:
                new_affine[mz][3] = -rsz*nsz/2.0
        new_affine[3][3] = 1.0

        src_img = nibabel.Nifti1Image(source.get_data(), new_affine, source.header)
        src_img.update_header()
        src_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_surface,
                                                        suffix='tmp_srcimg'))
        save_volume(src_img_file, src_img)
        source = load_volume(src_img_file)
        src_affine = source.affine
        src_header = source.header

        # create generic affine aligned with the orientation for the target
        new_affine = numpy.zeros((4,4))
        if ignore_header:
            new_affine[0][0] = rtx
            new_affine[1][1] = rty
            new_affine[2][2] = rtz
            new_affine[0][3] = -rtx*ntx/2.0
            new_affine[1][3] = -rty*nty/2.0
            new_affine[2][3] = -rtz*ntz/2.0
        else:
            #mx = numpy.argmax(numpy.abs(trg_affine[0][0:3]))
            #my = numpy.argmax(numpy.abs(trg_affine[1][0:3]))
            #mz = numpy.argmax(numpy.abs(trg_affine[2][0:3]))
            #new_affine[0][mx] = rtx*numpy.sign(trg_affine[0][mx])
            #new_affine[1][my] = rty*numpy.sign(trg_affine[1][my])
            #new_affine[2][mz] = rtz*numpy.sign(trg_affine[2][mz])
            #if (numpy.sign(trg_affine[0][mx])<0):
            #    new_affine[0][3] = rtx*ntx/2.0
            #else:
            #    new_affine[0][3] = -rtx*ntx/2.0
            #
            #if (numpy.sign(trg_affine[1][my])<0):
            #    new_affine[1][3] = rty*nty/2.0
            #else:
            #    new_affine[1][3] = -rty*nty/2.0
            #
            #if (numpy.sign(trg_affine[2][mz])<0):
            #    new_affine[2][3] = rtz*ntz/2.0
            #else:
            #    new_affine[2][3] = -rtz*ntz/2.0
            mx = numpy.argmax(numpy.abs([trg_affine[0][0],trg_affine[1][0],trg_affine[2][0]]))
            my = numpy.argmax(numpy.abs([trg_affine[0][1],trg_affine[1][1],trg_affine[2][1]]))
            mz = numpy.argmax(numpy.abs([trg_affine[0][2],trg_affine[1][2],trg_affine[2][2]]))
            #print('mx: '+str(mx)+', my: '+str(my)+', mz: '+str(mz))
            #print('rx: '+str(rtx)+', ry: '+str(rty)+', rz: '+str(rtz))
            new_affine[mx][0] = rtx*numpy.sign(trg_affine[mx][0])
            new_affine[my][1] = rty*numpy.sign(trg_affine[my][1])
            new_affine[mz][2] = rtz*numpy.sign(trg_affine[mz][2])
            if (numpy.sign(trg_affine[mx][0])<0):
                new_affine[mx][3] = rtx*ntx/2.0
            else:
                new_affine[mx][3] = -rtx*ntx/2.0

            if (numpy.sign(trg_affine[my][1])<0):
                new_affine[my][3] = rty*nty/2.0
            else:
                new_affine[my][3] = -rty*nty/2.0

            if (numpy.sign(trg_affine[mz][2])<0):
                new_affine[mz][3] = rtz*ntz/2.0
            else:
                new_affine[mz][3] = -rtz*ntz/2.0
        #if (numpy.sign(trg_affine[0][mx])<0): new_affine[mx][3] = rtx*ntx
        #if (numpy.sign(trg_affine[1][my])<0): new_affine[my][3] = rty*nty
        #if (numpy.sign(trg_affine[2][mz])<0): new_affine[mz][3] = rtz*ntz
        #new_affine[0][3] = ntx/2.0
        #new_affine[1][3] = nty/2.0
        #new_affine[2][3] = ntz/2.0
        new_affine[3][3] = 1.0
        #print("\nbefore: "+str(trg_affine))
        #print("\nafter: "+str(new_affine))
        trg_img = nibabel.Nifti1Image(target.get_data(), new_affine, target.header)
        trg_img.update_header()
        trg_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_surface,
                                                        suffix='tmp_trgimg'))
        save_volume(trg_img_file, trg_img)
        target = load_volume(trg_img_file)
        trg_affine = target.affine
        trg_header = target.header

    if mask_zero:
        # create and save temporary masks
        trg_mask_data = (target.get_data()!=0)*(target.get_data()!=2.0*max_dist)
        trg_mask = nibabel.Nifti1Image(trg_mask_data, target.affine, target.header)
        trg_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_surface,
                                                            suffix='tmp_trgmask'))
        save_volume(trg_mask_file, trg_mask)

        src_mask_data = (source.get_data()!=0)*(source.get_data()!=2.0*max_dist)
        src_mask = nibabel.Nifti1Image(src_mask_data, source.affine, source.header)
        src_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_surface,
                                                            suffix='tmp_srcmask'))
        save_volume(src_mask_file, src_mask)

    # run the main ANTS software: here we directly build the command line call
    reg = 'antsRegistration --collapse-output-transforms 1 --dimensionality 3' \
            +' --initialize-transforms-per-stage 0 --interpolation Linear'

     # add a prefix to avoid multiple names?
    prefix = _fname_4saving(module=__name__,file_name=file_name,
                            rootfile=source_surface,
                            suffix='tmp_syn')
    prefix = os.path.basename(prefix)
    prefix = prefix.split(".")[0]
    #reg.inputs.output_transform_prefix = prefix
    reg = reg+' --output '+prefix

    if mask_zero:
        reg = reg+' --masks ['+trg_mask_file+', '+src_mask_file+']'

    srcfiles = []
    trgfiles = []

    print("registering "+source.get_filename()+"\n to "+target.get_filename())
    srcfiles.append(source.get_filename())
    trgfiles.append(target.get_filename())

    weight = 1.0/len(srcfiles)

    # set parameters for all the different types of transformations
    if run_rigid is True:
        reg = reg + ' --transform Rigid[0.1]'
        for idx,img in enumerate(srcfiles):
            reg = reg + ' --metric MeanSquares['+trgfiles[idx]+', '+srcfiles[idx] \
                        +', '+'{:.3f}'.format(weight)+', 0, Random, 0.3 ]'

        reg = reg + ' --convergence ['+str(rigid_iterations)+'x' \
                    +str(rigid_iterations)+'x'+str(rigid_iterations)  \
                    +', '+str(convergence)+', 5 ]'

        reg = reg + ' --smoothing-sigmas 4.0x2.0x0.0'
        reg = reg + ' --shrink-factors 16x4x1'
        reg = reg + ' --use-histogram-matching 0'
        #reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_affine is True:
        reg = reg + ' --transform Affine[0.1]'
        for idx,img in enumerate(srcfiles):
            reg = reg + ' --metric MeanSquares['+trgfiles[idx]+', '+srcfiles[idx] \
                        +', '+'{:.3f}'.format(weight)+', 0, Random, 0.3 ]'

        reg = reg + ' --convergence ['+str(affine_iterations)+'x' \
                    +str(affine_iterations)+'x'+str(affine_iterations)  \
                    +', '+str(convergence)+', 5 ]'

        reg = reg + ' --smoothing-sigmas 4.0x2.0x0.0'
        reg = reg + ' --shrink-factors 16x4x1'
        reg = reg + ' --use-histogram-matching 0'
        #reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_syn is True:
        if regularization == 'Low': syn_param = [0.1, 1.0, 0.0]
        elif regularization == 'Medium': syn_param = [0.1, 3.0, 0.0]
        elif regularization == 'High': syn_param = [0.2, 4.0, 3.0]
        else: syn_param = [0.1, 3.0, 0.0]

        reg = reg + ' --transform SyN'+str(syn_param)
        if (cost_function=='Demons'):
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric Demons['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 4, Random, 0.3 ]'
        else:
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric MeanSquares['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 0, Random, 0.3 ]'

        reg = reg + ' --convergence ['+str(coarse_iterations)+'x' \
                    +str(coarse_iterations)+'x'+str(medium_iterations)+'x' \
                    +str(medium_iterations)+'x'  \
                    +str(fine_iterations)+', '+str(convergence)+', 5 ]'

        reg = reg + ' --smoothing-sigmas 9.0x6.0x3.0x1.0x0.0'
        reg = reg + ' --shrink-factors 16x8x4x2x1'
        reg = reg + ' --use-histogram-matching 0'
        #reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_rigid is False and run_affine is False and run_syn is False:
        reg = reg + ' --transform Rigid[0.1]'
        for idx,img in enumerate(srcfiles):
            reg = reg + ' --metric CC['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 5, Random, 0.3 ]'
        reg = reg + ' --convergence [ 0x0x0, 1.0, 2 ]'
        reg = reg + ' --smoothing-sigmas 3.0x2.0x1.0'
        reg = reg + ' --shrink-factors 4x2x1'
        reg = reg + ' --use-histogram-matching 0'
        #reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    reg = reg + ' --write-composite-transform 0'

    # run the ANTs command directly
    print(reg)
    try:
        subprocess.check_output(reg, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+str(e.returncode)+')\n Output: '+str(e.output)
        raise subprocess.CalledProcessError(msg)

    # output file names
    results = sorted(glob(prefix+'*'))
    forward = []
    flag = []
    for res in results:
        if res.endswith('GenericAffine.mat'):
            forward.append(res)
            flag.append(False)
        elif res.endswith('Warp.nii.gz') and not res.endswith('InverseWarp.nii.gz'):
            forward.append(res)
            flag.append(False)

    #print('forward transforms: '+str(forward))

    inverse = []
    linear = []
    for res in results[::-1]:
        if res.endswith('GenericAffine.mat'):
            inverse.append(res)
            linear.append(True)
        elif res.endswith('InverseWarp.nii.gz'):
            inverse.append(res)
            linear.append(False)

    #print('inverse transforms: '+str(inverse))

    #transform input (for checking)
#    src_at = 'antsApplyTransforms --dimensionality 3 --input-image-type 3'
#    src_at = src_at+' --input '+source.get_filename()
#    src_at = src_at+' --reference-image '+target.get_filename()
#    src_at = src_at+' --interpolation Linear'
#    for idx,transform in enumerate(forward):
#        if flag[idx]:
#            src_at = src_at+' --transform ['+transform+', 1]'
#        else:
#            src_at = src_at+' --transform ['+transform+', 0]'
#    src_at = src_at+' --output '+mapping_file
#
#    print(src_at)
#    try:
#        subprocess.check_output(src_at, shell=True, stderr=subprocess.STDOUT)
#    except subprocess.CalledProcessError as e:
#        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
#        raise subprocess.CalledProcessError(msg)

    # Create forward coordinate mapping
    src_coord = numpy.zeros((nsx,nsy,nsz,3))
    src_coord[:,:,:,0] = numpy.expand_dims(numpy.expand_dims(numpy.array(range(nsx)),1),2) \
                        *numpy.ones((1,nsy,1))*numpy.ones((1,1,nsz))
    src_coord[:,:,:,1] = numpy.ones((nsx,1,1))*numpy.expand_dims(numpy.expand_dims(numpy.array(range(nsy)),0),2) \
                        *numpy.ones((1,1,nsz))
    src_coord[:,:,:,2] = numpy.ones((nsx,1,1))*numpy.ones((1,nsy,1)) \
                        *numpy.expand_dims(numpy.expand_dims(numpy.array(range(nsz)),0),1)
    src_map = nibabel.Nifti1Image(src_coord, source.affine, source.header)
    src_map_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_surface,
                                                        suffix='tmp_srccoord'))
    save_volume(src_map_file, src_map)

    src_at = 'antsApplyTransforms --dimensionality 3 --input-image-type 3'
    src_at = src_at+' --input '+src_map.get_filename()
    src_at = src_at+' --reference-image '+target.get_filename()
    src_at = src_at+' --interpolation Linear'
    for idx,transform in enumerate(forward):
        if flag[idx]:
            src_at = src_at+' --transform ['+transform+', 1]'
        else:
            src_at = src_at+' --transform ['+transform+', 0]'
    src_at = src_at+' --output '+mapping_file

    print(src_at)
    try:
        subprocess.check_output(src_at, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
        raise subprocess.CalledProcessError(msg)

    # uncrop if needed
    if crop:
        orig = load_volume(target_surface)
        nx = orig.header.get_data_shape()[X]
        ny = orig.header.get_data_shape()[Y]
        nz = orig.header.get_data_shape()[Z]
        coord = -numpy.ones((nx,ny,nz,3))
        mapping = load_volume(mapping_file).get_data()
        coord[trg_xmin:trg_xmax+1, trg_ymin:trg_ymax+1, trg_zmin:trg_zmax+1, 0] = mapping[:,:,:,0] + src_xmin
        coord[trg_xmin:trg_xmax+1, trg_ymin:trg_ymax+1, trg_zmin:trg_zmax+1, 1] = mapping[:,:,:,1] + src_ymin
        coord[trg_xmin:trg_xmax+1, trg_ymin:trg_ymax+1, trg_zmin:trg_zmax+1, 2] = mapping[:,:,:,2] + src_zmin
        coord_img = nibabel.Nifti1Image(coord, orig.affine, orig.header)
        save_volume(mapping_file, coord_img)        

    # Create backward coordinate mapping
    trg_coord = numpy.zeros((ntx,nty,ntz,3))
    trg_coord[:,:,:,0] = numpy.expand_dims(numpy.expand_dims(numpy.array(range(ntx)),1),2) \
                        *numpy.ones((1,nty,1))*numpy.ones((1,1,ntz))
    trg_coord[:,:,:,1] = numpy.ones((ntx,1,1))*numpy.expand_dims(numpy.expand_dims(numpy.array(range(nty)),0),2) \
                        *numpy.ones((1,1,ntz))
    trg_coord[:,:,:,2] = numpy.ones((ntx,1,1))*numpy.ones((1,nty,1)) \
                        *numpy.expand_dims(numpy.expand_dims(numpy.array(range(ntz)),0),1)
    trg_map = nibabel.Nifti1Image(trg_coord, target.affine, target.header)
    trg_map_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_surface,
                                                        suffix='tmp_trgcoord'))
    save_volume(trg_map_file, trg_map)

    trg_at = 'antsApplyTransforms --dimensionality 3 --input-image-type 3'
    trg_at = trg_at+' --input '+trg_map.get_filename()
    trg_at = trg_at+' --reference-image '+source.get_filename()
    trg_at = trg_at+' --interpolation Linear'
    for idx,transform in enumerate(inverse):
        if linear[idx]:
            trg_at = trg_at+' --transform ['+transform+', 1]'
        else:
            trg_at = trg_at+' --transform ['+transform+', 0]'
    trg_at = trg_at+' --output '+inverse_mapping_file

    print(trg_at)
    try:
        subprocess.check_output(trg_at, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
        raise subprocess.CalledProcessError(msg)

    # uncrop if needed
    if crop:
        orig = load_volume(source_surface)
        nx = orig.header.get_data_shape()[X]
        ny = orig.header.get_data_shape()[Y]
        nz = orig.header.get_data_shape()[Z]
        coord = -numpy.ones((nx,ny,nz,3))
        mapping = load_volume(inverse_mapping_file).get_data()
        coord[src_xmin:src_xmax+1, src_ymin:src_ymax+1, src_zmin:src_zmax+1, 0] = mapping[:,:,:,0] + trg_xmin
        coord[src_xmin:src_xmax+1, src_ymin:src_ymax+1, src_zmin:src_zmax+1, 1] = mapping[:,:,:,1] + trg_ymin
        coord[src_xmin:src_xmax+1, src_ymin:src_ymax+1, src_zmin:src_zmax+1, 2] = mapping[:,:,:,2] + trg_zmin
        coord_img = nibabel.Nifti1Image(coord, orig.affine, orig.header)
        save_volume(inverse_mapping_file, coord_img)        

    # clean-up intermediate files
#    if os.path.exists(src_map_file): os.remove(src_map_file)
#    if os.path.exists(trg_map_file): os.remove(trg_map_file)
#    if os.path.exists(src_img_file): os.remove(src_img_file)
#    if os.path.exists(trg_img_file): os.remove(trg_img_file)
#    if mask_zero:
#        if os.path.exists(src_mask_file): os.remove(src_mask_file)
#        if os.path.exists(trg_mask_file): os.remove(trg_mask_file)

    for name in forward:
        if os.path.exists(name): os.remove(name)
    for name in inverse:
        if os.path.exists(name): os.remove(name)

    # if ignoring header and/or affine, must paste back the correct headers
    if ignore_affine or ignore_header:
        mapping = load_volume(mapping_file)
        save_volume(mapping_file, nibabel.Nifti1Image(mapping.get_data(), orig_trg_aff, orig_trg_hdr))
        inverse = load_volume(inverse_mapping_file)
        save_volume(inverse_mapping_file, nibabel.Nifti1Image(inverse.get_data(), orig_src_aff, orig_src_hdr))

    if not save_data:
        # collect saved outputs
        output = {'mapping': load_volume(mapping_file),
                  'inverse': load_volume(inverse_mapping_file)}

        # remove output files if *not* saved
        if os.path.exists(mapping_file): os.remove(mapping_file)
        if os.path.exists(inverse_mapping_file): os.remove(inverse_mapping_file)

        return output
    else:
        # collect saved outputs
        output = {'mapping': mapping_file,
                  'inverse': inverse_mapping_file}

        return output
