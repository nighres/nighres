# basic dependencies
import os
import sys
import subprocess
from glob import glob
import math

# main dependencies: numpy, nibabel
import numpy as np
import nibabel as nb

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

def embedded_antsreg(source_image, target_image,
                    run_rigid=False,
                    rigid_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,
					ignore_affine=False, ignore_header=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Embedded ANTS Registration

    Runs the rigid and/or Symmetric Normalization (SyN) algorithm of ANTs and
    formats the output deformations into voxel coordinate mappings as used in
    CBSTools registration and transformation routines.

    Parameters
    ----------
    source_image: niimg
        Image to register
    target_image: niimg
        Reference image to match
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
        Threshold for convergence, can make the algorithm very slow
        (default is convergence)
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

        * transformed_source (niimg): Deformed source image (_ants-def)
        * mapping (niimg): Coordinate mapping from source to target (_ants-map)
        * inverse (niimg): Inverse coordinate mapping from target to source
          (_ants-invmap)

    Notes
    ----------
    Port of the CBSTools Java module by Pierre-Louis Bazin. The main algorithm
    is part of the ANTs software by Brian Avants and colleagues [1]_. The
    interfacing with ANTs is performed through Nipype [2]_. Parameters have
    been set to values commonly found in neuroimaging scripts online, but not
    necessarily optimal.

    References
    ----------
    .. [1] Avants et al (2008), Symmetric diffeomorphic
       image registration with cross-correlation: evaluating automated labeling
       of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
    .. [2] Gorgolewski et al (2011) Nipype: a flexible, lightweight and
       extensible neuroimaging data processing framework in python. Front
       Neuroinform 5. doi:10.3389/fninf.2011.00013
    """

    # just overloading the multi-channel version
    return embedded_antsreg_multi([source_image], [target_image],
                    run_rigid, rigid_iterations, run_affine, affine_iterations,
                    run_syn, coarse_iterations, medium_iterations, fine_iterations,
					cost_function, interpolation, regularization, convergence,
					mask_zero, ignore_affine, ignore_header,
					save_data, overwrite, output_dir, file_name)


def embedded_antsreg_2d(source_image, target_image,
                    run_rigid=False,
                    rigid_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
                    scaling_factor=32,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,
					ignore_affine=False, ignore_header=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Embedded ANTS Registration 2D

    Runs the rigid and/or Symmetric Normalization (SyN) algorithm of ANTs and
    formats the output deformations into voxel coordinate mappings as used in
    CBSTools registration and transformation routines.

    Parameters
    ----------
    source_image: niimg
        Image to register
    target_image: niimg
        Reference image to match
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
    convergence: flaot
        Threshold for convergence, can make the algorithm very slow
        (default is convergence)
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

        * transformed_source (niimg): Deformed source image (_ants-def)
        * mapping (niimg): Coordinate mapping from source to target (_ants-map)
        * inverse (niimg): Inverse coordinate mapping from target to source
          (_ants-invmap)

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

    print('\nEmbedded ANTs Registration 2D')
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

     # filenames needed for intermediate results
    output_dir = _output_dir_4saving(output_dir, source_image)

    transformed_source_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_image,
                               suffix='ants-def'))

    mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_image,
                               suffix='ants-map'))

    inverse_mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_image,
                               suffix='ants-invmap'))
    if save_data:
        if overwrite is False \
            and os.path.isfile(transformed_source_file) \
            and os.path.isfile(mapping_file) \
            and os.path.isfile(inverse_mapping_file) :

            print("skip computation (use existing results)")
            output = {'transformed_source': transformed_source_file,
                      'mapping': mapping_file,
                      'inverse': inverse_mapping_file}
            return output


    # load and get dimensions and resolution from input images
    source = load_volume(source_image)
    src_affine = source.affine
    src_header = source.header
    nsx = source.header.get_data_shape()[X]
    nsy = source.header.get_data_shape()[Y]
    nsz = 1
    rsx = source.header.get_zooms()[X]
    rsy = source.header.get_zooms()[Y]
    rsz = 1

    target = load_volume(target_image)
    trg_affine = target.affine
    trg_header = target.header
    ntx = target.header.get_data_shape()[X]
    nty = target.header.get_data_shape()[Y]
    ntz = 1
    rtx = target.header.get_zooms()[X]
    rty = target.header.get_zooms()[Y]
    rtz = 1

    # in case the affine transformations are not to be trusted: make them equal
    if ignore_affine or ignore_header:
        mx = np.argmax(np.abs(src_affine[0][0:3]))
        my = np.argmax(np.abs(src_affine[1][0:3]))
        mz = np.argmax(np.abs(src_affine[2][0:3]))
        new_affine = np.zeros((4,4))
        if ignore_header:
            new_affine[0][0] = rsx
            new_affine[1][1] = rsy
            new_affine[2][2] = rsz
            new_affine[0][3] = -rsx*nsx/2.0
            new_affine[1][3] = -rsy*nsy/2.0
            new_affine[2][3] = -rsz*nsz/2.0
        else:
            new_affine[0][mx] = rsx*np.sign(src_affine[0][mx])
            new_affine[1][my] = rsy*np.sign(src_affine[1][my])
            new_affine[2][mz] = rsz*np.sign(src_affine[2][mz])
            if (np.sign(src_affine[0][mx])<0):
                new_affine[0][3] = rsx*nsx/2.0
            else:
                new_affine[0][3] = -rsx*nsx/2.0

            if (np.sign(src_affine[1][my])<0):
                new_affine[1][3] = rsy*nsy/2.0
            else:
                new_affine[1][3] = -rsy*nsy/2.0

            if (np.sign(src_affine[2][mz])<0):
                new_affine[2][3] = rsz*nsz/2.0
            else:
                new_affine[2][3] = -rsz*nsz/2.0
        #if (np.sign(src_affine[0][mx])<0): new_affine[mx][3] = rsx*nsx
        #if (np.sign(src_affine[1][my])<0): new_affine[my][3] = rsy*nsy
        #if (np.sign(src_affine[2][mz])<0): new_affine[mz][3] = rsz*nsz
        #new_affine[0][3] = nsx/2.0
        #new_affine[1][3] = nsy/2.0
        #new_affine[2][3] = nsz/2.0
        new_affine[3][3] = 1.0

        src_img = nb.Nifti1Image(source.get_data(), new_affine, source.header)
        src_img.update_header()
        src_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srcimg'))
        save_volume(src_img_file, src_img)
        source = load_volume(src_img_file)
        src_affine = source.affine
        src_header = source.header

        # create generic affine aligned with the orientation for the target
        mx = np.argmax(np.abs(trg_affine[0][0:3]))
        my = np.argmax(np.abs(trg_affine[1][0:3]))
        mz = np.argmax(np.abs(trg_affine[2][0:3]))
        new_affine = np.zeros((4,4))
        if ignore_header:
            new_affine[0][0] = rtx
            new_affine[1][1] = rty
            new_affine[2][2] = rtz
            new_affine[0][3] = -rtx*ntx/2.0
            new_affine[1][3] = -rty*nty/2.0
            new_affine[2][3] = -rtz*ntz/2.0
        else:
            new_affine[0][mx] = rtx*np.sign(trg_affine[0][mx])
            new_affine[1][my] = rty*np.sign(trg_affine[1][my])
            new_affine[2][mz] = rtz*np.sign(trg_affine[2][mz])
            if (np.sign(trg_affine[0][mx])<0):
                new_affine[0][3] = rtx*ntx/2.0
            else:
                new_affine[0][3] = -rtx*ntx/2.0

            if (np.sign(trg_affine[1][my])<0):
                new_affine[1][3] = rty*nty/2.0
            else:
                new_affine[1][3] = -rty*nty/2.0

            if (np.sign(trg_affine[2][mz])<0):
                new_affine[2][3] = rtz*ntz/2.0
            else:
                new_affine[2][3] = -rtz*ntz/2.0
        #if (np.sign(trg_affine[0][mx])<0): new_affine[mx][3] = rtx*ntx
        #if (np.sign(trg_affine[1][my])<0): new_affine[my][3] = rty*nty
        #if (np.sign(trg_affine[2][mz])<0): new_affine[mz][3] = rtz*ntz
        #new_affine[0][3] = ntx/2.0
        #new_affine[1][3] = nty/2.0
        #new_affine[2][3] = ntz/2.0
        new_affine[3][3] = 1.0

        trg_img = nb.Nifti1Image(target.get_data(), new_affine, target.header)
        trg_img.update_header()
        trg_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgimg'))
        save_volume(trg_img_file, trg_img)
        target = load_volume(trg_img_file)
        trg_affine = target.affine
        trg_header = target.header

    # build coordinate mapping matrices and save them to disk
    src_coordX = np.zeros((nsx,nsy))
    src_coordY = np.zeros((nsx,nsy))
    trg_coordX = np.zeros((ntx,nty))
    trg_coordY = np.zeros((ntx,nty))
    for x in range(nsx):
        for y in range(nsy):
            src_coordX[x,y] = x
            src_coordY[x,y] = y
    src_mapX = nb.Nifti1Image(src_coordX, source.affine, source.header)
    src_mapX_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordX'))
    save_volume(src_mapX_file, src_mapX)
    src_mapY = nb.Nifti1Image(src_coordY, source.affine, source.header)
    src_mapY_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordY'))
    save_volume(src_mapY_file, src_mapY)

    for x in range(ntx):
        for y in range(nty):
            trg_coordX[x,y] = x
            trg_coordY[x,y] = y
    trg_mapX = nb.Nifti1Image(trg_coordX, target.affine, target.header)
    trg_mapX_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgcoordX'))
    save_volume(trg_mapX_file, trg_mapX)
    trg_mapY = nb.Nifti1Image(trg_coordY, target.affine, target.header)
    trg_mapY_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgcoordY'))
    save_volume(trg_mapY_file, trg_mapY)

    if mask_zero:
        # create and save temporary masks
        trg_mask_data = (target.get_data()!=0)
        trg_mask = nb.Nifti1Image(trg_mask_data, target.affine, target.header)
        trg_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_image,
                                                            suffix='tmp_trgmask'))
        save_volume(trg_mask_file, trg_mask)

        src_mask_data = (source.get_data()!=0)
        src_mask = nb.Nifti1Image(src_mask_data, source.affine, source.header)
        src_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_image,
                                                            suffix='tmp_srcmask'))
        save_volume(src_mask_file, src_mask)

    # run the main ANTS software: here we directly build the command line call
    reg = 'antsRegistration --collapse-output-transforms 1 --dimensionality 2' \
            +' --initialize-transforms-per-stage 0 --interpolation Linear'

     # add a prefix to avoid multiple names?
    prefix = _fname_4saving(module=__name__,file_name=file_name,
                            rootfile=source_image,
                            suffix='tmp_syn')
    prefix = os.path.basename(prefix)
    prefix = prefix.split(".")[0]
    reg = reg+' --output '+prefix

    if mask_zero:
        reg = reg+' --masks ['+trg_mask_file+', '+src_mask_file+']'

    srcfile = source.get_filename()
    trgfile = target.get_filename()

    # figure out the number of scales, going with a factor of two
    n_scales = math.ceil(math.log(scaling_factor)/math.log(2.0))
    iter_rigid = str(rigid_iterations)
    iter_affine = str(affine_iterations)
    iter_syn = str(coarse_iterations)
    smooth = str(scaling_factor)
    shrink = str(scaling_factor)
    for n in range(n_scales):
        iter_rigid = iter_rigid+'x'+str(rigid_iterations)
        iter_affine = iter_affine+'x'+str(affine_iterations)
        if n<n_scales/2: iter_syn = iter_syn+'x'+str(coarse_iterations)
        elif n<n_scales-1: iter_syn = iter_syn+'x'+str(medium_iterations)
        else: iter_syn = iter_syn+'x'+str(fine_iterations)
        smooth = smooth+'x'+str(scaling_factor/math.pow(2.0,n+1))
        shrink = shrink+'x'+str(math.ceil(scaling_factor/math.pow(2.0,n+1)))

    # set parameters for all the different types of transformations
    if run_rigid is True:
        reg = reg + ' --transform Rigid[0.1]'
        if (cost_function=='CrossCorrelation'):
            reg = reg + ' --metric CC['+trgfile+', '+srcfile \
                            +', '+'1.0, 5, Random, 0.3 ]'
        else:
            reg = reg + ' --metric MI['+trgfile+', '+srcfile \
                            +', '+'1.0, 32, Random, 0.3 ]'

        reg = reg + ' --convergence ['+iter_rigid+', '+str(convergence)+', 10 ]'

        reg = reg + ' --smoothing-sigmas '+smooth
        reg = reg + ' --shrink-factors '+shrink
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_affine is True:
        reg = reg + ' --transform Affine[0.1]'
        if (cost_function=='CrossCorrelation'):
            reg = reg + ' --metric CC['+trgfile+', '+srcfile \
                            +', '+'1.0, 5, Random, 0.3 ]'
        else:
            reg = reg + ' --metric MI['+trgfile+', '+srcfile \
                            +', '+'1.0, 32, Random, 0.3 ]'

        reg = reg + ' --convergence ['+iter_affine+', '+str(convergence)+', 10 ]'

        reg = reg + ' --smoothing-sigmas '+smooth
        reg = reg + ' --shrink-factors '+shrink
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_syn is True:
        if regularization == 'Low': syn_param = [0.2, 1.0, 0.0]
        elif regularization == 'Medium': syn_param = [0.2, 3.0, 0.0]
        elif regularization == 'High': syn_param = [0.2, 4.0, 3.0]
        else: syn_param = [0.2, 3.0, 0.0]

        reg = reg + ' --transform SyN'+str(syn_param)
        if (cost_function=='CrossCorrelation'):
            reg = reg + ' --metric CC['+trgfile+', '+srcfile \
                            +', '+'1.0, 5, Random, 0.3 ]'
        else:
            reg = reg + ' --metric MI['+trgfile+', '+srcfile \
                            +', '+'1.0, 32, Random, 0.3 ]'

        reg = reg + ' --convergence ['+iter_syn+', '+str(convergence)+', 5 ]'

        reg = reg + ' --smoothing-sigmas '+smooth
        reg = reg + ' --shrink-factors '+shrink
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_rigid is False and run_affine is False and run_syn is False:
        reg = reg + ' --transform Rigid[0.1]'
        reg = reg + ' --metric CC['+trgfile+', '+srcfile \
                            +', '+'1.0, 5, Random, 0.3 ]'
        reg = reg + ' --convergence [ 0, 1.0, 2 ]'
        reg = reg + ' --smoothing-sigmas 1.0'
        reg = reg + ' --shrink-factors 1'
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

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

    # Transforms the moving image
    at = 'antsApplyTransforms --dimensionality 2 --input-image-type 0'
    at = at+' --input '+source.get_filename()
    at = at+' --reference-image '+target.get_filename()
    at = at+' --interpolation '+interpolation
    for idx,transform in enumerate(forward):
        if flag[idx]:
            at = at+' --transform ['+transform+', 1]'
        else:
            at = at+' --transform ['+transform+', 0]'
    at = at+' --output '+transformed_source_file

    print(at)
    try:
        subprocess.check_output(at, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
        raise subprocess.CalledProcessError(msg)

    # Create coordinate mappings
    src_at = 'antsApplyTransforms --dimensionality 2 --input-image-type 0'
    src_at = src_at+' --input '+src_mapX.get_filename()
    src_at = src_at+' --reference-image '+target.get_filename()
    src_at = src_at+' --interpolation Linear'
    for idx,transform in enumerate(forward):
        if flag[idx]:
            src_at = src_at+' --transform ['+transform+', 1]'
        else:
            src_at = src_at+' --transform ['+transform+', 0]'
#    src_at = src_at+' --output '+mapping_file
    src_mapX_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordX_map'))
    src_at = src_at+' --output '+src_mapX_trans

    print(src_at)
    try:
        subprocess.check_output(src_at, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
        raise subprocess.CalledProcessError(msg)

    src_at = 'antsApplyTransforms --dimensionality 2 --input-image-type 0'
    src_at = src_at+' --input '+src_mapY.get_filename()
    src_at = src_at+' --reference-image '+target.get_filename()
    src_at = src_at+' --interpolation Linear'
    for idx,transform in enumerate(forward):
        if flag[idx]:
            src_at = src_at+' --transform ['+transform+', 1]'
        else:
            src_at = src_at+' --transform ['+transform+', 0]'
#    src_at = src_at+' --output '+mapping_file
    src_mapY_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordY_map'))
    src_at = src_at+' --output '+src_mapY_trans

    print(src_at)
    try:
        subprocess.check_output(src_at, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
        raise subprocess.CalledProcessError(msg)

    # combine X,Y mappings
    mapX = load_volume(src_mapX_trans).get_data()
    mapY = load_volume(src_mapY_trans).get_data()
    src_map = np.stack((mapX,mapY),axis=-1)
    mapping = nb.Nifti1Image(src_map, target.affine, target.header)
    save_volume(mapping_file, mapping)


    trans_mapping = []

    trg_at = 'antsApplyTransforms --dimensionality 2 --input-image-type 0'
    trg_at = trg_at+' --input '+trg_mapX.get_filename()
    trg_at = trg_at+' --reference-image '+source.get_filename()
    trg_at = trg_at+' --interpolation Linear'
    for idx,transform in enumerate(inverse):
        if linear[idx]:
            trg_at = trg_at+' --transform ['+transform+', 1]'
        else:
            trg_at = trg_at+' --transform ['+transform+', 0]'
#    trg_at = trg_at+' --output '+inverse_mapping_file
    trg_mapX_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordX_map'))
    trg_at = trg_at+' --output '+trg_mapX_trans

    print(trg_at)
    try:
        subprocess.check_output(trg_at, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
        raise subprocess.CalledProcessError(msg)

    trg_at = 'antsApplyTransforms --dimensionality 2 --input-image-type 0'
    trg_at = trg_at+' --input '+trg_mapY.get_filename()
    trg_at = trg_at+' --reference-image '+source.get_filename()
    trg_at = trg_at+' --interpolation Linear'
    for idx,transform in enumerate(inverse):
        if linear[idx]:
            trg_at = trg_at+' --transform ['+transform+', 1]'
        else:
            trg_at = trg_at+' --transform ['+transform+', 0]'
#    trg_at = trg_at+' --output '+inverse_mapping_file
    trg_mapY_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordY_map'))
    trg_at = trg_at+' --output '+trg_mapY_trans

    print(trg_at)
    try:
        subprocess.check_output(trg_at, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
        raise subprocess.CalledProcessError(msg)

    # combine X,Y mappings
    mapX = load_volume(trg_mapX_trans).get_data()
    mapY = load_volume(trg_mapY_trans).get_data()
    trg_map = np.stack((mapX,mapY),axis=-1)
    inverse_mapping = nb.Nifti1Image(trg_map, source.affine, source.header)
    save_volume(inverse_mapping_file, inverse_mapping)

    # pad coordinate mapping outside the image? hopefully not needed...

    # clean-up intermediate files
    if os.path.exists(src_mapX_file): os.remove(src_mapX_file)
    if os.path.exists(src_mapY_file): os.remove(src_mapY_file)
    if os.path.exists(trg_mapX_file): os.remove(trg_mapX_file)
    if os.path.exists(trg_mapY_file): os.remove(trg_mapY_file)
    if os.path.exists(src_mapX_trans): os.remove(src_mapX_trans)
    if os.path.exists(src_mapY_trans): os.remove(src_mapY_trans)
    if os.path.exists(trg_mapX_trans): os.remove(trg_mapX_trans)
    if os.path.exists(trg_mapY_trans): os.remove(trg_mapY_trans)
    if ignore_affine or ignore_header:
        if os.path.exists(src_img_file): os.remove(src_img_file)
        if os.path.exists(trg_img_file): os.remove(trg_img_file)

    for name in forward:
        if os.path.exists(name): os.remove(name)
    for name in inverse:
        if os.path.exists(name): os.remove(name)

    if not save_data:
        # collect saved outputs
        output = {'transformed_source': load_volume(transformed_source_file),
              'mapping': load_volume(mapping_file),
              'inverse': load_volume(inverse_mapping_file)}

        # remove output files if *not* saved
        if os.path.exists(transformed_source_file): os.remove(transformed_source_file)
        if os.path.exists(mapping_file): os.remove(mapping_file)
        if os.path.exists(inverse_mapping_file): os.remove(inverse_mapping_file)

        return output
    else:
        # collect saved outputs
        output = {'transformed_source': transformed_source_file,
              'mapping': mapping_file,
              'inverse': inverse_mapping_file}

        return output

def embedded_antsreg_2d_multi(source_images, target_images,
                    run_rigid=False,
                    rigid_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
                    scaling_factor=32,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,
					ignore_affine=False, ignore_orient=False, ignore_res=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Embedded ANTS Registration 2D Multi-contrasts

    Runs the rigid and/or Symmetric Normalization (SyN) algorithm of ANTs and
    formats the output deformations into voxel coordinate mappings as used in
    CBSTools registration and transformation routines. Uses all input contrasts
    with equal weights.

    Parameters
    ----------
    source_images: [niimg]
        Images to register
    target_images: [niimg]
        Reference images to match
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
    convergence: flaot
        Threshold for convergence, can make the algorithm very slow
        (default is convergence)
    ignore_affine: bool
        Ignore the affine matrix information extracted from the image header
        (default is False)
    ignore_orient: bool
        Ignore the orientation information and affine matrix information
        extracted from the image header (default is False)
    ignore_res: bool
        Ignore the resolution information extracted from the image header
        (default is False)
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

        * transformed_source (niimg): Deformed source image (_ants-def)
        * mapping (niimg): Coordinate mapping from source to target (_ants-map)
        * inverse (niimg): Inverse coordinate mapping from target to source
          (_ants-invmap)

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

    print('\nEmbedded ANTs Registration 2D Multi-contrasts')
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

     # filenames needed for intermediate results
    output_dir = _output_dir_4saving(output_dir, source_images[0])

    transformed_source_files = []
    for idx,source_image in enumerate(source_images):
        transformed_source_files.append(os.path.join(output_dir,
                                    _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='ants-def'+str(idx))))

    mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='ants-map'))

    inverse_mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='ants-invmap'))
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


    # load and get dimensions and resolution from input images
    sources = []
    targets = []
    src_img_files = []
    trg_img_files = []
    for idx,img in enumerate(source_images):
        source = load_volume(source_images[idx])
        src_affine = source.affine
        src_header = source.header
        nsx = source.header.get_data_shape()[X]
        nsy = source.header.get_data_shape()[Y]
        nsz = 1
        rsx = source.header.get_zooms()[X]
        rsy = source.header.get_zooms()[Y]
        rsz = 1.0

        orig_src_aff = source.affine
        orig_src_hdr = source.header

        target = load_volume(target_images[idx])
        trg_affine = target.affine
        trg_header = target.header
        ntx = target.header.get_data_shape()[X]
        nty = target.header.get_data_shape()[Y]
        ntz = 1
        rtx = target.header.get_zooms()[X]
        rty = target.header.get_zooms()[Y]
        rtz = 1.0

        orig_trg_aff = target.affine
        orig_trg_hdr = target.header

        # in case the affine transformations are not to be trusted: make them equal
        if ignore_affine or ignore_orient or ignore_res:
            mx = np.argmax(np.abs(src_affine[0][0:3]))
            my = np.argmax(np.abs(src_affine[1][0:3]))
            mz = np.argmax(np.abs(src_affine[2][0:3]))
            new_affine = np.zeros((4,4))
            if ignore_res:
                new_affine[0][:] = src_affine[0][:]/rsx
                new_affine[1][:] = src_affine[1][:]/rsy
                new_affine[2][:] = src_affine[2][:]/rsz
                rsx = 1.0
                rsy = 1.0
                rsz = 1.0

            if ignore_orient:
                new_affine[0][0] = rsx
                new_affine[1][1] = rsy
                new_affine[2][2] = rsz
                new_affine[0][3] = -rsx*nsx/2.0
                new_affine[1][3] = -rsy*nsy/2.0
                new_affine[2][3] = -rsz*nsz/2.0
            elif ignore_affine:
                new_affine[0][mx] = rsx*np.sign(src_affine[0][mx])
                new_affine[1][my] = rsy*np.sign(src_affine[1][my])
                new_affine[2][mz] = rsz*np.sign(src_affine[2][mz])
                if (np.sign(src_affine[0][mx])<0):
                    new_affine[0][3] = rsx*nsx/2.0
                else:
                    new_affine[0][3] = -rsx*nsx/2.0

                if (np.sign(src_affine[1][my])<0):
                    new_affine[1][3] = rsy*nsy/2.0
                else:
                    new_affine[1][3] = -rsy*nsy/2.0

                if (np.sign(src_affine[2][mz])<0):
                    new_affine[2][3] = rsz*nsz/2.0
                else:
                    new_affine[2][3] = -rsz*nsz/2.0
            #if (np.sign(src_affine[0][mx])<0): new_affine[mx][3] = rsx*nsx
            #if (np.sign(src_affine[1][my])<0): new_affine[my][3] = rsy*nsy
            #if (np.sign(src_affine[2][mz])<0): new_affine[mz][3] = rsz*nsz
            #new_affine[0][3] = nsx/2.0
            #new_affine[1][3] = nsy/2.0
            #new_affine[2][3] = nsz/2.0
            new_affine[3][3] = 1.0

            src_img = nb.Nifti1Image(source.get_data(), new_affine, source.header)
            src_img.update_header()
            src_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_srcimg'+str(idx)))
            save_volume(src_img_file, src_img)
            source = load_volume(src_img_file)
            src_affine = source.affine
            src_header = source.header
            src_img_files.append(src_img_file)

            # create generic affine aligned with the orientation for the target
            mx = np.argmax(np.abs(trg_affine[0][0:3]))
            my = np.argmax(np.abs(trg_affine[1][0:3]))
            mz = np.argmax(np.abs(trg_affine[2][0:3]))
            new_affine = np.zeros((4,4))
            if ignore_res:
                new_affine[0][:] = trg_affine[0][:]/rtx
                new_affine[1][:] = trg_affine[1][:]/rty
                new_affine[2][:] = trg_affine[2][:]/rtz
                rtx = 1.0
                rty = 1.0
                rtz = 1.0

            if ignore_orient:
                new_affine[0][0] = rtx
                new_affine[1][1] = rty
                new_affine[2][2] = rtz
                new_affine[0][3] = -rtx*ntx/2.0
                new_affine[1][3] = -rty*nty/2.0
                new_affine[2][3] = -rtz*ntz/2.0
            elif ignore_affine:
                new_affine[0][mx] = rtx*np.sign(trg_affine[0][mx])
                new_affine[1][my] = rty*np.sign(trg_affine[1][my])
                new_affine[2][mz] = rtz*np.sign(trg_affine[2][mz])
                if (np.sign(trg_affine[0][mx])<0):
                    new_affine[0][3] = rtx*ntx/2.0
                else:
                    new_affine[0][3] = -rtx*ntx/2.0

                if (np.sign(trg_affine[1][my])<0):
                    new_affine[1][3] = rty*nty/2.0
                else:
                    new_affine[1][3] = -rty*nty/2.0

                if (np.sign(trg_affine[2][mz])<0):
                    new_affine[2][3] = rtz*ntz/2.0
                else:
                    new_affine[2][3] = -rtz*ntz/2.0
            #if (np.sign(trg_affine[0][mx])<0): new_affine[mx][3] = rtx*ntx
            #if (np.sign(trg_affine[1][my])<0): new_affine[my][3] = rty*nty
            #if (np.sign(trg_affine[2][mz])<0): new_affine[mz][3] = rtz*ntz
            #new_affine[0][3] = ntx/2.0
            #new_affine[1][3] = nty/2.0
            #new_affine[2][3] = ntz/2.0
            new_affine[3][3] = 1.0

            trg_img = nb.Nifti1Image(target.get_data(), new_affine, target.header)
            trg_img.update_header()
            trg_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_trgimg'+str(idx)))
            save_volume(trg_img_file, trg_img)
            target = load_volume(trg_img_file)
            trg_affine = target.affine
            trg_header = target.header
            trg_img_files.append(trg_img_file)

        sources.append(source)
        targets.append(target)

    # build coordinate mapping matrices and save them to disk
    src_coordX = np.zeros((nsx,nsy))
    src_coordY = np.zeros((nsx,nsy))
    trg_coordX = np.zeros((ntx,nty))
    trg_coordY = np.zeros((ntx,nty))
    for x in range(nsx):
        for y in range(nsy):
            src_coordX[x,y] = x
            src_coordY[x,y] = y
    src_mapX = nb.Nifti1Image(src_coordX, source.affine, source.header)
    src_mapX_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_srccoordX'))
    save_volume(src_mapX_file, src_mapX)
    src_mapY = nb.Nifti1Image(src_coordY, source.affine, source.header)
    src_mapY_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_srccoordY'))
    save_volume(src_mapY_file, src_mapY)

    for x in range(ntx):
        for y in range(nty):
            trg_coordX[x,y] = x
            trg_coordY[x,y] = y
    trg_mapX = nb.Nifti1Image(trg_coordX, target.affine, target.header)
    trg_mapX_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_trgcoordX'))
    save_volume(trg_mapX_file, trg_mapX)
    trg_mapY = nb.Nifti1Image(trg_coordY, target.affine, target.header)
    trg_mapY_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_trgcoordY'))
    save_volume(trg_mapY_file, trg_mapY)

    if mask_zero:
        # create and save temporary masks
        target = targets[0]
        trg_mask_data = (target.get_data()!=0)
        trg_mask = nb.Nifti1Image(trg_mask_data, target.affine, target.header)
        trg_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_trgmask'))
        save_volume(trg_mask_file, trg_mask)

        source = sources[0]
        src_mask_data = (source.get_data()!=0)
        src_mask = nb.Nifti1Image(src_mask_data, source.affine, source.header)
        src_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_srcmask'))
        save_volume(src_mask_file, src_mask)

    # run the main ANTS software: here we directly build the command line call
    reg = 'antsRegistration --collapse-output-transforms 1 --dimensionality 2' \
            +' --initialize-transforms-per-stage 0 --interpolation Linear'

     # add a prefix to avoid multiple names?
    prefix = _fname_4saving(module=__name__,file_name=file_name,
                            rootfile=source_images[0],
                            suffix='tmp_syn')
    prefix = os.path.basename(prefix)
    prefix = prefix.split(".")[0]
    reg = reg+' --output '+prefix

    if mask_zero:
        reg = reg+' --masks ['+trg_mask_file+', '+src_mask_file+']'

    srcfiles = []
    trgfiles = []
    for idx,img in enumerate(sources):
        print("registering "+sources[idx].get_filename()+"\n to "+targets[idx].get_filename())
        srcfiles.append(sources[idx].get_filename())
        trgfiles.append(targets[idx].get_filename())

    weight = 1.0/len(srcfiles)

    # figure out the number of scales, going with a factor of two
    n_scales = math.ceil(math.log(scaling_factor)/math.log(2.0))
    iter_rigid = str(rigid_iterations)
    iter_affine = str(affine_iterations)
    iter_syn = str(coarse_iterations)
    smooth = str(scaling_factor)
    shrink = str(scaling_factor)
    for n in range(n_scales):
        iter_rigid = iter_rigid+'x'+str(rigid_iterations)
        iter_affine = iter_affine+'x'+str(affine_iterations)
        if n<n_scales/2: iter_syn = iter_syn+'x'+str(coarse_iterations)
        elif n<n_scales-1: iter_syn = iter_syn+'x'+str(medium_iterations)
        else: iter_syn = iter_syn+'x'+str(fine_iterations)
        smooth = smooth+'x'+str(scaling_factor/math.pow(2.0,n+1))
        shrink = shrink+'x'+str(math.ceil(scaling_factor/math.pow(2.0,n+1)))

    # set parameters for all the different types of transformations
    if run_rigid is True:
        reg = reg + ' --transform Rigid[0.1]'
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric CC['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 5, Random, 0.3 ]'
        else:
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric MI['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 32, Random, 0.3 ]'

        reg = reg + ' --convergence ['+iter_rigid+', '+str(convergence)+', 10 ]'

        reg = reg + ' --smoothing-sigmas '+smooth
        reg = reg + ' --shrink-factors '+shrink
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_affine is True:
        reg = reg + ' --transform Affine[0.1]'
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric CC['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 5, Random, 0.3 ]'
        else:
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric MI['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 32, Random, 0.3 ]'

        reg = reg + ' --convergence ['+iter_affine+', '+str(convergence)+', 10 ]'

        reg = reg + ' --smoothing-sigmas '+smooth
        reg = reg + ' --shrink-factors '+shrink
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_syn is True:
        if regularization == 'Low': syn_param = [0.2, 1.0, 0.0]
        elif regularization == 'Medium': syn_param = [0.2, 3.0, 0.0]
        elif regularization == 'High': syn_param = [0.2, 4.0, 3.0]
        else: syn_param = [0.2, 3.0, 0.0]

        reg = reg + ' --transform SyN'+str(syn_param)
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric CC['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 5, Random, 0.3 ]'
        else:
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric MI['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 32, Random, 0.3 ]'

        reg = reg + ' --convergence ['+iter_syn+', '+str(convergence)+', 5 ]'

        reg = reg + ' --smoothing-sigmas '+smooth
        reg = reg + ' --shrink-factors '+shrink
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_rigid is False and run_affine is False and run_syn is False:
        reg = reg + ' --transform Rigid[0.1]'
        for idx,img in enumerate(srcfiles):
            reg = reg + ' --metric CC['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 5, Random, 0.3 ]'
        reg = reg + ' --convergence [ 0, 1.0, 2 ]'
        reg = reg + ' --smoothing-sigmas 1.0'
        reg = reg + ' --shrink-factors 1'
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

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

    # Transforms the moving image
    for idx,source in enumerate(sources):
        at = 'antsApplyTransforms --dimensionality 2 --input-image-type 0'
        at = at+' --input '+sources[idx].get_filename()
        at = at+' --reference-image '+targets[idx].get_filename()
        at = at+' --interpolation '+interpolation
        for idx2,transform in enumerate(forward):
            if flag[idx2]:
                at = at+' --transform ['+transform+', 1]'
            else:
                at = at+' --transform ['+transform+', 0]'
        at = at+' --output '+transformed_source_files[idx]

        print(at)
        try:
            subprocess.check_output(at, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
            raise subprocess.CalledProcessError(msg)

    # Create coordinate mappings
    src_at = 'antsApplyTransforms --dimensionality 2 --input-image-type 0'
    src_at = src_at+' --input '+src_mapX.get_filename()
    src_at = src_at+' --reference-image '+target.get_filename()
    src_at = src_at+' --interpolation Linear'
    for idx,transform in enumerate(forward):
        if flag[idx]:
            src_at = src_at+' --transform ['+transform+', 1]'
        else:
            src_at = src_at+' --transform ['+transform+', 0]'
#    src_at = src_at+' --output '+mapping_file
    src_mapX_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordX_map'))
    src_at = src_at+' --output '+src_mapX_trans

    print(src_at)
    try:
        subprocess.check_output(src_at, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
        raise subprocess.CalledProcessError(msg)

    src_at = 'antsApplyTransforms --dimensionality 2 --input-image-type 0'
    src_at = src_at+' --input '+src_mapY.get_filename()
    src_at = src_at+' --reference-image '+target.get_filename()
    src_at = src_at+' --interpolation Linear'
    for idx,transform in enumerate(forward):
        if flag[idx]:
            src_at = src_at+' --transform ['+transform+', 1]'
        else:
            src_at = src_at+' --transform ['+transform+', 0]'
#    src_at = src_at+' --output '+mapping_file
    src_mapY_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordY_map'))
    src_at = src_at+' --output '+src_mapY_trans

    print(src_at)
    try:
        subprocess.check_output(src_at, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
        raise subprocess.CalledProcessError(msg)

    # combine X,Y mappings
    mapX = load_volume(src_mapX_trans).get_data()
    mapY = load_volume(src_mapY_trans).get_data()
    src_map = np.stack((mapX,mapY),axis=-1)
    mapping = nb.Nifti1Image(src_map, target.affine, target.header)
    save_volume(mapping_file, mapping)


    trans_mapping = []

    trg_at = 'antsApplyTransforms --dimensionality 2 --input-image-type 0'
    trg_at = trg_at+' --input '+trg_mapX.get_filename()
    trg_at = trg_at+' --reference-image '+source.get_filename()
    trg_at = trg_at+' --interpolation Linear'
    for idx,transform in enumerate(inverse):
        if linear[idx]:
            trg_at = trg_at+' --transform ['+transform+', 1]'
        else:
            trg_at = trg_at+' --transform ['+transform+', 0]'
#    trg_at = trg_at+' --output '+inverse_mapping_file
    trg_mapX_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordX_map'))
    trg_at = trg_at+' --output '+trg_mapX_trans

    print(trg_at)
    try:
        subprocess.check_output(trg_at, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
        raise subprocess.CalledProcessError(msg)

    trg_at = 'antsApplyTransforms --dimensionality 2 --input-image-type 0'
    trg_at = trg_at+' --input '+trg_mapY.get_filename()
    trg_at = trg_at+' --reference-image '+source.get_filename()
    trg_at = trg_at+' --interpolation Linear'
    for idx,transform in enumerate(inverse):
        if linear[idx]:
            trg_at = trg_at+' --transform ['+transform+', 1]'
        else:
            trg_at = trg_at+' --transform ['+transform+', 0]'
#    trg_at = trg_at+' --output '+inverse_mapping_file
    trg_mapY_trans = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoordY_map'))
    trg_at = trg_at+' --output '+trg_mapY_trans

    print(trg_at)
    try:
        subprocess.check_output(trg_at, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
        raise subprocess.CalledProcessError(msg)

    # combine X,Y mappings
    mapX = load_volume(trg_mapX_trans).get_data()
    mapY = load_volume(trg_mapY_trans).get_data()
    trg_map = np.stack((mapX,mapY),axis=-1)
    inverse_mapping = nb.Nifti1Image(trg_map, source.affine, source.header)
    save_volume(inverse_mapping_file, inverse_mapping)

    # pad coordinate mapping outside the image? hopefully not needed...

    # clean-up intermediate files
    if os.path.exists(src_mapX_file): os.remove(src_mapX_file)
    if os.path.exists(src_mapY_file): os.remove(src_mapY_file)
    if os.path.exists(trg_mapX_file): os.remove(trg_mapX_file)
    if os.path.exists(trg_mapY_file): os.remove(trg_mapY_file)
    if os.path.exists(src_mapX_trans): os.remove(src_mapX_trans)
    if os.path.exists(src_mapY_trans): os.remove(src_mapY_trans)
    if os.path.exists(trg_mapX_trans): os.remove(trg_mapX_trans)
    if os.path.exists(trg_mapY_trans): os.remove(trg_mapY_trans)
    if ignore_affine or ignore_orient or ignore_res:
        for src_img_file in src_img_files:
            if os.path.exists(src_img_file): os.remove(src_img_file)
        for trg_img_file in trg_img_files:
            if os.path.exists(trg_img_file): os.remove(trg_img_file)

    for name in forward:
        if os.path.exists(name): os.remove(name)
    for name in inverse:
        if os.path.exists(name): os.remove(name)

    # if ignoring header and/or affine, must paste back the correct headers
    if ignore_affine or ignore_orient or ignore_res:
        mapping = load_volume(mapping_file)
        save_volume(mapping_file, nb.Nifti1Image(mapping.get_data(), orig_trg_aff, orig_trg_hdr))
        inverse = load_volume(inverse_mapping_file)
        save_volume(inverse_mapping_file, nb.Nifti1Image(inverse.get_data(), orig_src_aff, orig_src_hdr))
        for trans_file in transformed_source_files:
            trans = load_volume(trans_file)
            save_volume(trans_file, nb.Nifti1Image(trans.get_data(), orig_trg_aff, orig_trg_hdr))

    if not save_data:
        # collect saved outputs
        transformed = []
        for trans_file in transformed_source_files:
            transformed.append(load_volume(trans_file))
        output = {'transformed_sources': transformed,
              'transformed_source': transformed[0],
              'mapping': load_volume(mapping_file),
              'inverse': load_volume(inverse_mapping_file)}

        # remove output files if *not* saved
        for idx,trans_image in enumerate(transformed_source_files):
            if os.path.exists(trans_image): os.remove(trans_image)
        if os.path.exists(mapping_file): os.remove(mapping_file)
        if os.path.exists(inverse_mapping_file): os.remove(inverse_mapping_file)

        return output
    else:
        # collect saved outputs
        transformed = []
        for trans_file in transformed_source_files:
            transformed.append(trans_file)
        output = {'transformed_sources': transformed,
              'transformed_source': transformed[0],
              'mapping': mapping_file,
              'inverse': inverse_mapping_file}

        return output

def embedded_antsreg_multi(source_images, target_images,
                    run_rigid=True,
                    rigid_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					regularization='High',
					convergence=1e-6,
					mask_zero=False,
					ignore_affine=False, ignore_header=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Embedded ANTS Registration Multi-contrasts

    Runs the rigid and/or Symmetric Normalization (SyN) algorithm of ANTs and
    formats the output deformations into voxel coordinate mappings as used in
    CBSTools registration and transformation routines. Uses all input contrasts
    with equal weights.

    Parameters
    ----------
    source_images: [niimg]
        Image list to register
    target_images: [niimg]
        Reference image list to match
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

        * transformed_source ([niimg]): Deformed source image list (_ants_def0,1,...)
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

    print('\nEmbedded ANTs Registration Multi-contrasts')
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
    output_dir = _output_dir_4saving(output_dir, source_images[0])

    transformed_source_files = []
    for idx,source_image in enumerate(source_images):
        transformed_source_files.append(os.path.join(output_dir,
                                    _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='ants-def'+str(idx))))

    mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='ants-map'))

    inverse_mapping_file = os.path.join(output_dir,
                    _fname_4saving(module=__name__,file_name=file_name,
                               rootfile=source_images[0],
                               suffix='ants-invmap'))
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

    # load and get dimensions and resolution from input images
    sources = []
    targets = []
    for idx,img in enumerate(source_images):
        source = load_volume(source_images[idx])
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

        target = load_volume(target_images[idx])
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
            new_affine = np.zeros((4,4))
            if ignore_header:
                new_affine[0][0] = rsx
                new_affine[1][1] = rsy
                new_affine[2][2] = rsz
                new_affine[0][3] = -rsx*nsx/2.0
                new_affine[1][3] = -rsy*nsy/2.0
                new_affine[2][3] = -rsz*nsz/2.0
            else:
                #mx = np.argmax(np.abs(src_affine[0][0:3]))
                #my = np.argmax(np.abs(src_affine[1][0:3]))
                #mz = np.argmax(np.abs(src_affine[2][0:3]))
                #new_affine[0][mx] = rsx*np.sign(src_affine[0][mx])
                #new_affine[1][my] = rsy*np.sign(src_affine[1][my])
                #new_affine[2][mz] = rsz*np.sign(src_affine[2][mz])
                #if (np.sign(src_affine[0][mx])<0):
                #    new_affine[0][3] = rsx*nsx/2.0
                #else:
                #    new_affine[0][3] = -rsx*nsx/2.0
                #
                #if (np.sign(src_affine[1][my])<0):
                #    new_affine[1][3] = rsy*nsy/2.0
                #else:
                #    new_affine[1][3] = -rsy*nsy/2.0
                #
                #if (np.sign(src_affine[2][mz])<0):
                #    new_affine[2][3] = rsz*nsz/2.0
                #else:
                #    new_affine[2][3] = -rsz*nsz/2.0
                mx = np.argmax(np.abs([src_affine[0][0],src_affine[1][0],src_affine[2][0]]))
                my = np.argmax(np.abs([src_affine[0][1],src_affine[1][1],src_affine[2][1]]))
                mz = np.argmax(np.abs([src_affine[0][2],src_affine[1][2],src_affine[2][2]]))
                new_affine[mx][0] = rsx*np.sign(src_affine[mx][0])
                new_affine[my][1] = rsy*np.sign(src_affine[my][1])
                new_affine[mz][2] = rsz*np.sign(src_affine[mz][2])
                if (np.sign(src_affine[mx][0])<0):
                    new_affine[mx][3] = rsx*nsx/2.0
                else:
                    new_affine[mx][3] = -rsx*nsx/2.0

                if (np.sign(src_affine[my][1])<0):
                    new_affine[my][3] = rsy*nsy/2.0
                else:
                    new_affine[my][3] = -rsy*nsy/2.0

                if (np.sign(src_affine[mz][2])<0):
                    new_affine[mz][3] = rsz*nsz/2.0
                else:
                    new_affine[mz][3] = -rsz*nsz/2.0
            #if (np.sign(src_affine[0][mx])<0): new_affine[mx][3] = rsx*nsx
            #if (np.sign(src_affine[1][my])<0): new_affine[my][3] = rsy*nsy
            #if (np.sign(src_affine[2][mz])<0): new_affine[mz][3] = rsz*nsz
            #new_affine[0][3] = nsx/2.0
            #new_affine[1][3] = nsy/2.0
            #new_affine[2][3] = nsz/2.0
            new_affine[3][3] = 1.0

            src_img = nb.Nifti1Image(source.get_data(), new_affine, source.header)
            src_img.update_header()
            src_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_srcimg'+str(idx)))
            save_volume(src_img_file, src_img)
            source = load_volume(src_img_file)
            src_affine = source.affine
            src_header = source.header

            # create generic affine aligned with the orientation for the target
            new_affine = np.zeros((4,4))
            if ignore_header:
                new_affine[0][0] = rtx
                new_affine[1][1] = rty
                new_affine[2][2] = rtz
                new_affine[0][3] = -rtx*ntx/2.0
                new_affine[1][3] = -rty*nty/2.0
                new_affine[2][3] = -rtz*ntz/2.0
            else:
                #mx = np.argmax(np.abs(trg_affine[0][0:3]))
                #my = np.argmax(np.abs(trg_affine[1][0:3]))
                #mz = np.argmax(np.abs(trg_affine[2][0:3]))
                #new_affine[0][mx] = rtx*np.sign(trg_affine[0][mx])
                #new_affine[1][my] = rty*np.sign(trg_affine[1][my])
                #new_affine[2][mz] = rtz*np.sign(trg_affine[2][mz])
                #if (np.sign(trg_affine[0][mx])<0):
                #    new_affine[0][3] = rtx*ntx/2.0
                #else:
                #    new_affine[0][3] = -rtx*ntx/2.0
                #
                #if (np.sign(trg_affine[1][my])<0):
                #    new_affine[1][3] = rty*nty/2.0
                #else:
                #    new_affine[1][3] = -rty*nty/2.0
                #
                #if (np.sign(trg_affine[2][mz])<0):
                #    new_affine[2][3] = rtz*ntz/2.0
                #else:
                #    new_affine[2][3] = -rtz*ntz/2.0
                mx = np.argmax(np.abs([trg_affine[0][0],trg_affine[1][0],trg_affine[2][0]]))
                my = np.argmax(np.abs([trg_affine[0][1],trg_affine[1][1],trg_affine[2][1]]))
                mz = np.argmax(np.abs([trg_affine[0][2],trg_affine[1][2],trg_affine[2][2]]))
                #print('mx: '+str(mx)+', my: '+str(my)+', mz: '+str(mz))
                #print('rx: '+str(rtx)+', ry: '+str(rty)+', rz: '+str(rtz))
                new_affine[mx][0] = rtx*np.sign(trg_affine[mx][0])
                new_affine[my][1] = rty*np.sign(trg_affine[my][1])
                new_affine[mz][2] = rtz*np.sign(trg_affine[mz][2])
                if (np.sign(trg_affine[mx][0])<0):
                    new_affine[mx][3] = rtx*ntx/2.0
                else:
                    new_affine[mx][3] = -rtx*ntx/2.0

                if (np.sign(trg_affine[my][1])<0):
                    new_affine[my][3] = rty*nty/2.0
                else:
                    new_affine[my][3] = -rty*nty/2.0

                if (np.sign(trg_affine[mz][2])<0):
                    new_affine[mz][3] = rtz*ntz/2.0
                else:
                    new_affine[mz][3] = -rtz*ntz/2.0
            #if (np.sign(trg_affine[0][mx])<0): new_affine[mx][3] = rtx*ntx
            #if (np.sign(trg_affine[1][my])<0): new_affine[my][3] = rty*nty
            #if (np.sign(trg_affine[2][mz])<0): new_affine[mz][3] = rtz*ntz
            #new_affine[0][3] = ntx/2.0
            #new_affine[1][3] = nty/2.0
            #new_affine[2][3] = ntz/2.0
            new_affine[3][3] = 1.0
            #print("\nbefore: "+str(trg_affine))
            #print("\nafter: "+str(new_affine))
            trg_img = nb.Nifti1Image(target.get_data(), new_affine, target.header)
            trg_img.update_header()
            trg_img_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_trgimg'+str(idx)))
            save_volume(trg_img_file, trg_img)
            target = load_volume(trg_img_file)
            trg_affine = target.affine
            trg_header = target.header

        sources.append(source)
        targets.append(target)

    # build coordinate mapping matrices and save them to disk
    src_coord = np.zeros((nsx,nsy,nsz,3))
    trg_coord = np.zeros((ntx,nty,ntz,3))
    for x in range(nsx):
        for y in range(nsy):
            for z in range(nsz):
                src_coord[x,y,z,X] = x
                src_coord[x,y,z,Y] = y
                src_coord[x,y,z,Z] = z
    src_map = nb.Nifti1Image(src_coord, source.affine, source.header)
    src_map_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_srccoord'))
    save_volume(src_map_file, src_map)
    for x in range(ntx):
        for y in range(nty):
            for z in range(ntz):
                trg_coord[x,y,z,X] = x
                trg_coord[x,y,z,Y] = y
                trg_coord[x,y,z,Z] = z
    trg_map = nb.Nifti1Image(trg_coord, target.affine, target.header)
    trg_map_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_trgcoord'))
    save_volume(trg_map_file, trg_map)

    if mask_zero:
        # create and save temporary masks
        target = targets[0]
        trg_mask_data = (target.get_data()!=0)
        trg_mask = nb.Nifti1Image(trg_mask_data, target.affine, target.header)
        trg_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_trgmask'))
        save_volume(trg_mask_file, trg_mask)

        source = sources[0]
        src_mask_data = (source.get_data()!=0)
        src_mask = nb.Nifti1Image(src_mask_data, source.affine, source.header)
        src_mask_file = os.path.join(output_dir, _fname_4saving(module=__name__,file_name=file_name,
                                                            rootfile=source_images[0],
                                                            suffix='tmp_srcmask'))
        save_volume(src_mask_file, src_mask)

    # run the main ANTS software: here we directly build the command line call
    reg = 'antsRegistration --collapse-output-transforms 1 --dimensionality 3' \
            +' --initialize-transforms-per-stage 0 --interpolation Linear'

     # add a prefix to avoid multiple names?
    prefix = _fname_4saving(module=__name__,file_name=file_name,
                            rootfile=source_images[0],
                            suffix='tmp_syn')
    prefix = os.path.basename(prefix)
    prefix = prefix.split(".")[0]
    #reg.inputs.output_transform_prefix = prefix
    reg = reg+' --output '+prefix

    if mask_zero:
        reg = reg+' --masks ['+trg_mask_file+', '+src_mask_file+']'

    srcfiles = []
    trgfiles = []
    for idx,img in enumerate(sources):
        print("registering "+sources[idx].get_filename()+"\n to "+targets[idx].get_filename())
        srcfiles.append(sources[idx].get_filename())
        trgfiles.append(targets[idx].get_filename())

    weight = 1.0/len(srcfiles)

    # set parameters for all the different types of transformations
    if run_rigid is True:
        reg = reg + ' --transform Rigid[0.1]'
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric CC['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 5, Random, 0.3 ]'
        else:
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric MI['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 32, Random, 0.3 ]'

        reg = reg + ' --convergence ['+str(rigid_iterations)+'x' \
                    +str(rigid_iterations)+'x'+str(rigid_iterations)  \
                    +', '+str(convergence)+', 10 ]'

        reg = reg + ' --smoothing-sigmas 3.0x2.0x1.0'
        reg = reg + ' --shrink-factors 4x2x1'
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_affine is True:
        reg = reg + ' --transform Affine[0.1]'
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric CC['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 5, Random, 0.3 ]'
        else:
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric MI['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 32, Random, 0.3 ]'

        reg = reg + ' --convergence ['+str(affine_iterations)+'x' \
                    +str(affine_iterations)+'x'+str(affine_iterations)  \
                    +', '+str(convergence)+', 10 ]'

        reg = reg + ' --smoothing-sigmas 3.0x2.0x1.0'
        reg = reg + ' --shrink-factors 4x2x1'
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_syn is True:
        if regularization == 'Low': syn_param = [0.2, 1.0, 0.0]
        elif regularization == 'Medium': syn_param = [0.2, 3.0, 0.0]
        elif regularization == 'High': syn_param = [0.2, 4.0, 3.0]
        else: syn_param = [0.2, 3.0, 0.0]

        reg = reg + ' --transform SyN'+str(syn_param)
        if (cost_function=='CrossCorrelation'):
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric CC['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 5, Random, 0.3 ]'
        else:
            for idx,img in enumerate(srcfiles):
                reg = reg + ' --metric MI['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 32, Random, 0.3 ]'

        reg = reg + ' --convergence ['+str(coarse_iterations)+'x' \
                    +str(coarse_iterations)+'x'+str(medium_iterations)+'x'  \
                    +str(fine_iterations)+', '+str(convergence)+', 5 ]'

        reg = reg + ' --smoothing-sigmas 2.0x1.0x0.5x0.0'
        reg = reg + ' --shrink-factors 8x4x2x1'
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

    if run_rigid is False and run_affine is False and run_syn is False:
        reg = reg + ' --transform Rigid[0.1]'
        for idx,img in enumerate(srcfiles):
            reg = reg + ' --metric CC['+trgfiles[idx]+', '+srcfiles[idx] \
                            +', '+'{:.3f}'.format(weight)+', 5, Random, 0.3 ]'
        reg = reg + ' --convergence [ 0x0x0, 1.0, 2 ]'
        reg = reg + ' --smoothing-sigmas 3.0x2.0x1.0'
        reg = reg + ' --shrink-factors 4x2x1'
        reg = reg + ' --use-histogram-matching 0'
        reg = reg + ' --winsorize-image-intensities [ 0.001, 0.999 ]'

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

    # Transforms the moving image
    for idx,source in enumerate(sources):
        at = 'antsApplyTransforms --dimensionality 3 --input-image-type 0'
        at = at+' --input '+sources[idx].get_filename()
        at = at+' --reference-image '+targets[idx].get_filename()
        at = at+' --interpolation '+interpolation
        for idx2,transform in enumerate(forward):
            if flag[idx2]:
                at = at+' --transform ['+transform+', 1]'
            else:
                at = at+' --transform ['+transform+', 0]'
        at = at+' --output '+transformed_source_files[idx]

        print(at)
        try:
            subprocess.check_output(at, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
            raise subprocess.CalledProcessError(msg)

    # Create coordinate mappings
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
    trans_mapping = []

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

    # pad coordinate mapping outside the image? hopefully not needed...

    # clean-up intermediate files
    if os.path.exists(src_map_file): os.remove(src_map_file)
    if os.path.exists(trg_map_file): os.remove(trg_map_file)
    if ignore_affine or ignore_header:
        if os.path.exists(src_img_file): os.remove(src_img_file)
        if os.path.exists(trg_img_file): os.remove(trg_img_file)

    for name in forward:
        if os.path.exists(name): os.remove(name)
    for name in inverse:
        if os.path.exists(name): os.remove(name)

    # if ignoring header and/or affine, must paste back the correct headers
    if ignore_affine or ignore_header:
        mapping = load_volume(mapping_file)
        save_volume(mapping_file, nb.Nifti1Image(mapping.get_data(), orig_trg_aff, orig_trg_hdr))
        inverse = load_volume(inverse_mapping_file)
        save_volume(inverse_mapping_file, nb.Nifti1Image(inverse.get_data(), orig_src_aff, orig_src_hdr))
        for trans_file in transformed_source_files:
            trans = load_volume(trans_file)
            save_volume(trans_file, nb.Nifti1Image(trans.get_data(), orig_trg_aff, orig_trg_hdr))

    if not save_data:
        # collect saved outputs
        transformed = []
        for trans_file in transformed_source_files:
            transformed.append(load_volume(trans_file))
        output = {'transformed_sources': transformed,
              'transformed_source': transformed[0],
              'mapping': load_volume(mapping_file),
              'inverse': load_volume(inverse_mapping_file)}

        # remove output files if *not* saved
        for idx,trans_image in enumerate(transformed_source_files):
            if os.path.exists(trans_image): os.remove(trans_image)
        if os.path.exists(mapping_file): os.remove(mapping_file)
        if os.path.exists(inverse_mapping_file): os.remove(inverse_mapping_file)

        return output
    else:
        # collect saved outputs
        transformed = []
        for trans_file in transformed_source_files:
            transformed.append(trans_file)
        output = {'transformed_sources': transformed,
              'transformed_source': transformed[0],
              'mapping': mapping_file,
              'inverse': inverse_mapping_file}

        return output
