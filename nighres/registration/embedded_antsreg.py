# basic dependencies
import os
import sys
import subprocess
from glob import glob

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

    print('\nEmbedded ANTs Registration')

    # for external tools: nipype
    try:
        from nipype.interfaces.ants import Registration
        from nipype.interfaces.ants import ApplyTransforms
    except ImportError:
        print('Error: Nipype and/or ANTS could not be imported, they are required'
                +' in order to run this module. \n (aborting)')
        return None

    # make sure that saving related parameters are correct
    output_dir = _output_dir_4saving(output_dir, source_image) # needed for intermediate results
    if save_data:
        transformed_source_file = os.path.join(output_dir,
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='ants-def'))

        mapping_file = os.path.join(output_dir,
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='ants-map'))

        inverse_mapping_file = os.path.join(output_dir,
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='ants-invmap'))
        if overwrite is False \
            and os.path.isfile(transformed_source_file) \
            and os.path.isfile(mapping_file) \
            and os.path.isfile(inverse_mapping_file) :

            print("skip computation (use existing results)")
            output = {'transformed_source': load_volume(transformed_source_file),
                      'mapping': load_volume(mapping_file),
                      'inverse': load_volume(inverse_mapping_file)}
            return output


    # load and get dimensions and resolution from input images
    source = load_volume(source_image)
    src_affine = source.affine
    src_header = source.header
    nsx = source.header.get_data_shape()[X]
    nsy = source.header.get_data_shape()[Y]
    nsz = source.header.get_data_shape()[Z]
    rsx = source.header.get_zooms()[X]
    rsy = source.header.get_zooms()[Y]
    rsz = source.header.get_zooms()[Z]

    target = load_volume(target_image)
    trg_affine = target.affine
    trg_header = target.header
    ntx = target.header.get_data_shape()[X]
    nty = target.header.get_data_shape()[Y]
    ntz = target.header.get_data_shape()[Z]
    rtx = target.header.get_zooms()[X]
    rty = target.header.get_zooms()[Y]
    rtz = target.header.get_zooms()[Z]

    # in case the affine transformations are not to be trusted: make them equal
    if ignore_affine or ignore_header:
        # create generic affine aligned with the orientation for the source
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
        src_img_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
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

        trg_img = nb.Nifti1Image(target.get_data(), new_affine, target.header)
        trg_img.update_header()
        trg_img_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgimg'))
        save_volume(trg_img_file, trg_img)
        target = load_volume(trg_img_file)
        trg_affine = target.affine
        trg_header = target.header

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
    src_map_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoord'))
    save_volume(src_map_file, src_map)
    for x in range(ntx):
        for y in range(nty):
            for z in range(ntz):
                trg_coord[x,y,z,X] = x
                trg_coord[x,y,z,Y] = y
                trg_coord[x,y,z,Z] = z
    trg_map = nb.Nifti1Image(trg_coord, target.affine, target.header)
    trg_map_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgcoord'))
    save_volume(trg_map_file, trg_map)

    if mask_zero:
        # create and save temporary masks
        trg_mask_data = (target.get_data()!=0)
        trg_mask = nb.Nifti1Image(trg_mask_data, target.affine, target.header)
        trg_mask_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                            rootfile=source_image,
                                                            suffix='tmp_trgmask'))
        save_volume(trg_mask_file, trg_mask)
        
        src_mask_data = (source.get_data()!=0)
        src_mask = nb.Nifti1Image(src_mask_data, source.affine, source.header)
        src_mask_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                            rootfile=source_image,
                                                            suffix='tmp_srcmask'))
        save_volume(src_mask_file, src_mask)
        

    # run the main ANTS software: here we directly build the command line call
    reg = 'antsRegistration --collapse-output-transforms 1 --dimensionality 3' \
            +' --initialize-transforms-per-stage 0 --interpolation Linear'

     # add a prefix to avoid multiple names?
    prefix = _fname_4saving(file_name=file_name,
                            rootfile=source_image,
                            suffix='tmp_syn')
    prefix = os.path.basename(prefix)
    prefix = prefix.split(".")[0]
    #reg.inputs.output_transform_prefix = prefix
    reg = reg+' --output '+prefix

    reg.inputs.fixed_image = [target.get_filename()]
    reg.inputs.moving_image = [source.get_filename()]

    if mask_zero:
        reg.inputs.fixed_image_mask = trg_mask_file
        reg.inputs.moving_image_mask = src_mask_file
        

    print("registering "+source.get_filename()+"\n to "+target.get_filename())

    if run_syn is True:
        if regularization is 'Low': syn_param = (0.2, 1.0, 0.0)
        elif regularization is 'Medium': syn_param = (0.2, 3.0, 0.0)
        elif regularization is 'High': syn_param = (0.2, 4.0, 3.0)
        else: syn_param = (0.2, 3.0, 0.0)

    if run_rigid is True and run_affine is True and run_syn is True:
        reg.inputs.transforms = ['Rigid','Affine','SyN']
        reg.inputs.transform_parameters = [(0.1,), (0.1,), syn_param]
        reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                            rigid_iterations],
                                           [affine_iterations, affine_iterations,
                                            affine_iterations],
                                           [coarse_iterations, coarse_iterations,
                                            medium_iterations, fine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC', 'CC', 'CC']
            reg.inputs.metric_weight = [1.0, 1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [5, 5, 5]
        else :
            reg.inputs.metric = ['MI', 'MI', 'MI']
            reg.inputs.metric_weight = [1.0, 1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [32, 32, 32]
        reg.inputs.shrink_factors = [[4, 2, 1]] + [[4, 2, 1]] + [[8, 4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[3, 2, 1]] + [[2, 1, 0.5, 0]]
        reg.inputs.sampling_strategy = ['Random'] + ['Random'] + ['Random']
        reg.inputs.sampling_percentage = [0.3] + [0.3] + [0.3]
        reg.inputs.convergence_threshold = [convergence] + [convergence] + [convergence]
        reg.inputs.convergence_window_size = [10] + [10] + [5]
        reg.inputs.use_histogram_matching = [False] + [False] + [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is True and run_affine is False and run_syn is True:
        reg.inputs.transforms = ['Rigid','SyN']
        reg.inputs.transform_parameters = [(0.1,), syn_param]
        reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                            rigid_iterations],
                                           [coarse_iterations, coarse_iterations,
                                            medium_iterations, fine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC', 'CC']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [5, 5]
        else :
            reg.inputs.metric = ['MI', 'MI']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [32, 32]
        reg.inputs.shrink_factors = [[4, 2, 1]] + [[8, 4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[2, 1, 0.5, 0]]
        reg.inputs.sampling_strategy = ['Random'] + ['Random']
        reg.inputs.sampling_percentage = [0.3] + [0.3]
        reg.inputs.convergence_threshold = [convergence] + [convergence]
        reg.inputs.convergence_window_size = [10] + [5]
        reg.inputs.use_histogram_matching = [False] + [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is False and run_affine is True and run_syn is True:
        reg.inputs.transforms = ['Affine','SyN']
        reg.inputs.transform_parameters = [(0.1,), syn_param]
        reg.inputs.number_of_iterations = [[affine_iterations, affine_iterations,
                                            affine_iterations],
                                           [coarse_iterations, coarse_iterations,
                                            medium_iterations, fine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC', 'CC']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [5, 5]
        else :
            reg.inputs.metric = ['MI', 'MI']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [64, 64]
        reg.inputs.shrink_factors = [[4, 2, 1]] + [[8, 4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[2, 1, 0.5, 0]]
        reg.inputs.sampling_strategy = ['Random'] + ['Random']
        reg.inputs.sampling_percentage = [0.3] + [0.3]
        reg.inputs.convergence_threshold = [convergence] + [convergence]
        reg.inputs.convergence_window_size = [10] + [5]
        reg.inputs.use_histogram_matching = [False] + [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    if run_rigid is True and run_affine is True and run_syn is False:
        reg.inputs.transforms = ['Rigid','Affine']
        reg.inputs.transform_parameters = [(0.1,), (0.1,)]
        reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                            rigid_iterations],
                                           [affine_iterations, affine_iterations,
                                            affine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC', 'CC']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [5, 5]
        else :
            reg.inputs.metric = ['MI', 'MI']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [32, 32]
        reg.inputs.shrink_factors = [[4, 2, 1]] + [[4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[3, 2, 1]]
        reg.inputs.sampling_strategy = ['Random'] + ['Random']
        reg.inputs.sampling_percentage = [0.3] + [0.3]
        reg.inputs.convergence_threshold = [convergence] + [convergence]
        reg.inputs.convergence_window_size = [10] + [10]
        reg.inputs.use_histogram_matching = [False] + [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is True and run_affine is False and run_syn is False:
        reg.inputs.transforms = ['Rigid']
        reg.inputs.transform_parameters = [(0.1,)]
        reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                            rigid_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [5]
        else :
            reg.inputs.metric = ['MI']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [32]
        reg.inputs.shrink_factors = [[4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]]
        reg.inputs.sampling_strategy = ['Random']
        reg.inputs.sampling_percentage = [0.3]
        reg.inputs.convergence_threshold = [convergence]
        reg.inputs.convergence_window_size = [10]
        reg.inputs.use_histogram_matching = [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is False and run_affine is True and run_syn is False:
        reg.inputs.transforms = ['Affine']
        reg.inputs.transform_parameters = [(0.1,)]
        reg.inputs.number_of_iterations = [[affine_iterations, affine_iterations,
                                            affine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [5]
        else :
            reg.inputs.metric = ['MI']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [32]
        reg.inputs.shrink_factors = [[4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]]
        reg.inputs.sampling_strategy = ['Random']
        reg.inputs.sampling_percentage = [0.3]
        reg.inputs.convergence_threshold = [convergence]
        reg.inputs.convergence_window_size = [10]
        reg.inputs.use_histogram_matching = [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is False and run_affine is False and run_syn is True:
        reg.inputs.transforms = ['SyN']
        reg.inputs.transform_parameters = [syn_param]
        reg.inputs.number_of_iterations = [[coarse_iterations, coarse_iterations,
                                            medium_iterations, fine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [5]
        else :
            reg.inputs.metric = ['MI']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [32]
        reg.inputs.shrink_factors = [[8, 4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[2, 1, 0.5, 0]]
        reg.inputs.sampling_strategy = ['Random']
        reg.inputs.sampling_percentage = [0.3]
        reg.inputs.convergence_threshold = [convergence]
        reg.inputs.convergence_window_size = [10]
        reg.inputs.use_histogram_matching = [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is False and run_affine is False and run_syn is False:
        reg.inputs.transforms = ['Rigid']
        reg.inputs.transform_parameters = [(0.1,)]
        reg.inputs.number_of_iterations = [[0]]
        reg.inputs.metric = ['CC']
        reg.inputs.metric_weight = [1.0]
        reg.inputs.radius_or_number_of_bins = [5]
        reg.inputs.shrink_factors = [[1]]
        reg.inputs.smoothing_sigmas = [[1]]
        reg.inputs.args = '--float 0'



    print(reg.cmdline)
    result = reg.run()

    # Transforms the moving image
    at = ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = source.get_filename()
    at.inputs.reference_image = target.get_filename()
    at.inputs.interpolation = interpolation
    at.inputs.transforms = result.outputs.forward_transforms
    at.inputs.invert_transform_flags = result.outputs.forward_invert_flags
    transformed = at.run()

    # Create coordinate mappings
    src_at = ApplyTransforms()
    src_at.inputs.dimension = 3
    src_at.inputs.input_image_type = 3
    src_at.inputs.input_image = src_map.get_filename()
    src_at.inputs.reference_image = target.get_filename()
    src_at.inputs.interpolation = 'Linear'
    src_at.inputs.transforms = result.outputs.forward_transforms
    src_at.inputs.invert_transform_flags = result.outputs.forward_invert_flags
    mapping = src_at.run()

    trg_at = ApplyTransforms()
    trg_at.inputs.dimension = 3
    trg_at.inputs.input_image_type = 3
    trg_at.inputs.input_image = trg_map.get_filename()
    trg_at.inputs.reference_image = source.get_filename()
    trg_at.inputs.interpolation = 'Linear'
    trg_at.inputs.transforms = result.outputs.reverse_transforms
    trg_at.inputs.invert_transform_flags = result.outputs.reverse_invert_flags
    inverse = trg_at.run()

    # pad coordinate mapping outside the image? hopefully not needed...
    # unfortunately, yes needed: but we'll do that in apply_coordinate_mapping

    # collect outputs and potentially save
    transformed_img = nb.Nifti1Image(nb.load(transformed.outputs.output_image).get_data(),
                                    target.affine, target.header)
    mapping_img = nb.Nifti1Image(nb.load(mapping.outputs.output_image).get_data(),
                                    target.affine, target.header)
    inverse_img = nb.Nifti1Image(nb.load(inverse.outputs.output_image).get_data(),
                                    source.affine, source.header)

    outputs = {'transformed_source': transformed_img, 'mapping': mapping_img,
                'inverse': inverse_img}

    # clean-up intermediate files
    os.remove(src_map_file)
    os.remove(trg_map_file)
    if ignore_affine or ignore_header:
        os.remove(src_img_file)
        os.remove(trg_img_file)
    if mask_zero:
        os.remove(src_mask_file)
        os.remove(trg_mask_file)

    for name in result.outputs.forward_transforms:
        if os.path.exists(name): os.remove(name)
    for name in result.outputs.reverse_transforms:
        if os.path.exists(name): os.remove(name)
    os.remove(transformed.outputs.output_image)
    os.remove(mapping.outputs.output_image)
    os.remove(inverse.outputs.output_image)

    if save_data:
        save_volume(transformed_source_file, transformed_img)
        save_volume(mapping_file, mapping_img)
        save_volume(inverse_mapping_file, inverse_img)

    return outputs

def embedded_antsreg_2d(source_image, target_image,
                    run_rigid=False,
                    rigid_iterations=1000,
                    run_affine=False,
                    affine_iterations=1000,
                    run_syn=True,
                    coarse_iterations=40,
                    medium_iterations=50, fine_iterations=40,
					cost_function='MutualInformation',
					interpolation='NearestNeighbor',
					convergence=1e-6,
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
    is part of the ANTs software by Brian Avants and colleagues [1]_. The
    interfacing with ANTs is performed through Nipype [2]_. Parameters have been
    set to values commonly found in neuroimaging scripts online, but not
    necessarily optimal.

    References
    ----------
    .. [1] Avants et al (2008), Symmetric diffeomorphic
       image registration with cross-correlation: evaluating automated labeling
       of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
    .. [2] Gorgolewski et al (2011) Nipype: a flexible, lightweight and
       extensible neuroimaging data processing framework in python.
       Front Neuroinform 5. doi:10.3389/fninf.2011.00013
    """

    print('\nEmbedded ANTs Registration')

    # for external tools: nipype
    try:
        from nipype.interfaces.ants import Registration
        from nipype.interfaces.ants import ApplyTransforms
    except ImportError:
        print('Error: Nipype and/or ANTS could not be imported, they are required'
                +' in order to run this module. \n (aborting)')
        return None

    # make sure that saving related parameters are correct
    output_dir = _output_dir_4saving(output_dir, source_image) # needed for intermediate results
    if save_data:
        transformed_source_file = os.path.join(output_dir,
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='ants-def'))

        mapping_file = os.path.join(output_dir,
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='ants-map'))

        inverse_mapping_file = os.path.join(output_dir,
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='ants-invmap'))
        if overwrite is False \
            and os.path.isfile(transformed_source_file) \
            and os.path.isfile(mapping_file) \
            and os.path.isfile(inverse_mapping_file) :

            print("skip computation (use existing results)")
            output = {'transformed_source': load_volume(transformed_source_file),
                      'mapping': load_volume(mapping_file),
                      'inverse': load_volume(inverse_mapping_file)}
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
        src_img_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
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
        trg_img_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgimg'))
        save_volume(trg_img_file, trg_img)
        target = load_volume(trg_img_file)
        trg_affine = target.affine
        trg_header = target.header

    # build coordinate mapping matrices and save them to disk
    src_coord = np.zeros((nsx,nsy,2))
    trg_coord = np.zeros((ntx,nty,2))
    for x in range(nsx):
        for y in range(nsy):
            src_coord[x,y,X] = x
            src_coord[x,y,Y] = y
    src_map = nb.Nifti1Image(src_coord, source.affine, source.header)
    src_map_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoord'))
    save_volume(src_map_file, src_map)
    for x in range(ntx):
        for y in range(nty):
            trg_coord[x,y,X] = x
            trg_coord[x,y,Y] = y
    trg_map = nb.Nifti1Image(trg_coord, target.affine, target.header)
    trg_map_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_trgcoord'))
    save_volume(trg_map_file, trg_map)

    # run the main ANTS software
    reg = Registration()
    reg.inputs.dimension = 2

     # add a prefix to avoid multiple names?
    prefix = _fname_4saving(file_name=file_name,
                            rootfile=source_image,
                            suffix='tmp_syn')
    prefix = os.path.basename(prefix)
    prefix = prefix.split(".")[0]
    reg.inputs.output_transform_prefix = prefix
    reg.inputs.fixed_image = [target.get_filename()]
    reg.inputs.moving_image = [source.get_filename()]

    print("registering "+source.get_filename()+"\n to "+target.get_filename())

    if run_rigid is True and run_affine is True and run_syn is True:
        reg.inputs.transforms = ['Rigid','Affine','SyN']
        reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
        reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                            rigid_iterations],
                                           [affine_iterations, affine_iterations,
                                            affine_iterations],
                                           [coarse_iterations, coarse_iterations,
                                            medium_iterations, fine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC', 'CC', 'CC']
            reg.inputs.metric_weight = [1.0, 1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [5, 5, 5]
        else :
            reg.inputs.metric = ['MI', 'MI', 'MI']
            reg.inputs.metric_weight = [1.0, 1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [32, 32, 32]
        reg.inputs.shrink_factors = [[4, 2, 1]] + [[4, 2, 1]] + [[8, 4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[3, 2, 1]] + [[2, 1, 0.5, 0]]
        reg.inputs.sampling_strategy = ['Random'] + ['Random'] + ['Random']
        reg.inputs.sampling_percentage = [0.3] + [0.3] + [0.3]
        reg.inputs.convergence_threshold = [convergence] + [convergence] + [convergence]
        reg.inputs.convergence_window_size = [10] + [10] + [5]
        reg.inputs.use_histogram_matching = [False] + [False] + [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is True and run_affine is False and run_syn is True:
        reg.inputs.transforms = ['Rigid','SyN']
        reg.inputs.transform_parameters = [(0.1,), (0.2, 3.0, 0.0)]
        reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                            rigid_iterations],
                                           [coarse_iterations, coarse_iterations,
                                            medium_iterations, fine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC', 'CC']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [5, 5]
        else :
            reg.inputs.metric = ['MI', 'MI']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [32, 32]
        reg.inputs.shrink_factors = [[4, 2, 1]] + [[8, 4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[2, 1, 0.5, 0]]
        reg.inputs.sampling_strategy = ['Random'] + ['Random']
        reg.inputs.sampling_percentage = [0.3] + [0.3]
        reg.inputs.convergence_threshold = [convergence] + [convergence]
        reg.inputs.convergence_window_size = [10] + [5]
        reg.inputs.use_histogram_matching = [False] + [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is False and run_affine is True and run_syn is True:
        reg.inputs.transforms = ['Affine','SyN']
        reg.inputs.transform_parameters = [(0.1,), (0.2, 3.0, 0.0)]
        reg.inputs.number_of_iterations = [[affine_iterations, affine_iterations,
                                            affine_iterations],
                                           [coarse_iterations, coarse_iterations,
                                            medium_iterations, fine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC', 'CC']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [5, 5]
        else :
            reg.inputs.metric = ['MI', 'MI']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [64, 64]
        reg.inputs.shrink_factors = [[4, 2, 1]] + [[8, 4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[2, 1, 0.5, 0]]
        reg.inputs.sampling_strategy = ['Random'] + ['Random']
        reg.inputs.sampling_percentage = [0.3] + [0.3]
        reg.inputs.convergence_threshold = [convergence] + [convergence]
        reg.inputs.convergence_window_size = [10] + [5]
        reg.inputs.use_histogram_matching = [False] + [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    if run_rigid is True and run_affine is True and run_syn is False:
        reg.inputs.transforms = ['Rigid','Affine']
        reg.inputs.transform_parameters = [(0.1,), (0.1,)]
        reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                            rigid_iterations],
                                           [affine_iterations, affine_iterations,
                                            affine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC', 'CC']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [5, 5]
        else :
            reg.inputs.metric = ['MI', 'MI']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [32, 32]
        reg.inputs.shrink_factors = [[4, 2, 1]] + [[4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]] + [[3, 2, 1]]
        reg.inputs.sampling_strategy = ['Random'] + ['Random']
        reg.inputs.sampling_percentage = [0.3] + [0.3]
        reg.inputs.convergence_threshold = [convergence] + [convergence]
        reg.inputs.convergence_window_size = [10] + [10]
        reg.inputs.use_histogram_matching = [False] + [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is True and run_affine is False and run_syn is False:
        reg.inputs.transforms = ['Rigid']
        reg.inputs.transform_parameters = [(0.1,)]
        reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                            rigid_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [5]
        else :
            reg.inputs.metric = ['MI']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [32]
        reg.inputs.shrink_factors = [[4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]]
        reg.inputs.sampling_strategy = ['Random']
        reg.inputs.sampling_percentage = [0.3]
        reg.inputs.convergence_threshold = [convergence]
        reg.inputs.convergence_window_size = [10]
        reg.inputs.use_histogram_matching = [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is False and run_affine is True and run_syn is False:
        reg.inputs.transforms = ['Affine']
        reg.inputs.transform_parameters = [(0.1,)]
        reg.inputs.number_of_iterations = [[affine_iterations, affine_iterations,
                                            affine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [5]
        else :
            reg.inputs.metric = ['MI']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [32]
        reg.inputs.shrink_factors = [[4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[3, 2, 1]]
        reg.inputs.sampling_strategy = ['Random']
        reg.inputs.sampling_percentage = [0.3]
        reg.inputs.convergence_threshold = [convergence]
        reg.inputs.convergence_window_size = [10]
        reg.inputs.use_histogram_matching = [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is False and run_affine is False and run_syn is True:
        reg.inputs.transforms = ['SyN']
        reg.inputs.transform_parameters = [(0.2, 3.0, 0.0)]
        reg.inputs.number_of_iterations = [[coarse_iterations, coarse_iterations,
                                            medium_iterations, fine_iterations]]
        if (cost_function=='CrossCorrelation'):
            reg.inputs.metric = ['CC']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [5]
        else :
            reg.inputs.metric = ['MI']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [32]
        reg.inputs.shrink_factors = [[8, 4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[2, 1, 0.5, 0]]
        reg.inputs.sampling_strategy = ['Random']
        reg.inputs.sampling_percentage = [0.3]
        reg.inputs.convergence_threshold = [convergence]
        reg.inputs.convergence_window_size = [10]
        reg.inputs.use_histogram_matching = [False]
        reg.inputs.winsorize_lower_quantile = 0.001
        reg.inputs.winsorize_upper_quantile = 0.999

    elif run_rigid is False and run_affine is False and run_syn is False:
        reg.inputs.transforms = ['Rigid']
        reg.inputs.transform_parameters = [(0.1,)]
        reg.inputs.number_of_iterations = [[0]]
        reg.inputs.metric = ['CC']
        reg.inputs.metric_weight = [1.0]
        reg.inputs.radius_or_number_of_bins = [5]
        reg.inputs.shrink_factors = [[1]]
        reg.inputs.smoothing_sigmas = [[1]]



    print(reg.cmdline)
    result = reg.run()

    # Transforms the moving image
    at = ApplyTransforms()
    at.inputs.dimension = 2
    at.inputs.input_image = source.get_filename()
    at.inputs.reference_image = target.get_filename()
    at.inputs.interpolation = interpolation
    at.inputs.transforms = result.outputs.forward_transforms
    at.inputs.invert_transform_flags = result.outputs.forward_invert_flags
    print(at.cmdline)
    transformed = at.run()

    # Create coordinate mappings
    src_at = ApplyTransforms()
    src_at.inputs.dimension = 2
    src_at.inputs.input_image_type = 3
    src_at.inputs.input_image = src_map.get_filename()
    src_at.inputs.reference_image = target.get_filename()
    src_at.inputs.interpolation = 'Linear'
    src_at.inputs.transforms = result.outputs.forward_transforms
    src_at.inputs.invert_transform_flags = result.outputs.forward_invert_flags
    mapping = src_at.run()

    trg_at = ApplyTransforms()
    trg_at.inputs.dimension = 2
    trg_at.inputs.input_image_type = 3
    trg_at.inputs.input_image = trg_map.get_filename()
    trg_at.inputs.reference_image = source.get_filename()
    trg_at.inputs.interpolation = 'Linear'
    trg_at.inputs.transforms = result.outputs.reverse_transforms
    trg_at.inputs.invert_transform_flags = result.outputs.reverse_invert_flags
    inverse = trg_at.run()

    # pad coordinate mapping outside the image? hopefully not needed...

    # collect outputs and potentially save
    transformed_img = nb.Nifti1Image(nb.load(transformed.outputs.output_image).get_data(),
                                    target.affine, target.header)
    mapping_img = nb.Nifti1Image(nb.load(mapping.outputs.output_image).get_data(),
                                    target.affine, target.header)
    inverse_img = nb.Nifti1Image(nb.load(inverse.outputs.output_image).get_data(),
                                    source.affine, source.header)

    outputs = {'transformed_source': transformed_img, 'mapping': mapping_img,
                'inverse': inverse_img}

    # clean-up intermediate files
    os.remove(src_map_file)
    os.remove(trg_map_file)
    if ignore_affine or ignore_header:
        os.remove(src_img_file)
        os.remove(trg_img_file)

    for name in result.outputs.forward_transforms:
        if os.path.exists(name): os.remove(name)
    for name in result.outputs.reverse_transforms:
        if os.path.exists(name): os.remove(name)
    os.remove(transformed.outputs.output_image)
    os.remove(mapping.outputs.output_image)
    os.remove(inverse.outputs.output_image)

    if save_data:
        save_volume(transformed_source_file, transformed_img)
        save_volume(mapping_file, mapping_img)
        save_volume(inverse_mapping_file, inverse_img)

    return outputs

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
    is part of the ANTs software by Brian Avants and colleagues [1]_. The interfacing
    with ANTs is performed through Nipype [2]_. Parameters have been set to values
    commonly found in neuroimaging scripts online, but not necessarily optimal.

    References
    ----------
    .. [1] Avants et al (2008), Symmetric diffeomorphic 
       image registration with cross-correlation: evaluating automated labeling 
       of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
    .. [2] Gorgolewski et al (2011) Nipype: a flexible, lightweight and extensible 
       neuroimaging data processing framework in python. Front Neuroinform 5. 
       doi:10.3389/fninf.2011.00013
    """

    print('\nEmbedded ANTs Registration Multi-contrasts')

    # for external tools: nipype
    try:
        from nipype.interfaces.ants import Registration
        from nipype.interfaces.ants import ApplyTransforms
    except ImportError:
        print('Error: Nipype and/or ANTS could not be imported, they are required' 
                +'in order to run this module. \n (aborting)')
        return None

    # make sure that saving related parameters are correct
    output_dir = _output_dir_4saving(output_dir, source_images[0]) # needed for intermediate results
    if save_data:
        transformed_source_files = []
        for idx,source_image in enumerate(source_images):
            transformed_source_files.append(os.path.join(output_dir, 
                                        _fname_4saving(file_name=file_name,
                                       rootfile=source_image,
                                       suffix='ants-def'+str(idx))))

        mapping_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_images[0],
                                   suffix='ants-map'))

        inverse_mapping_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_images[0],
                                   suffix='ants-invmap'))
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
                    transformed.append(load_volume(trans_file)) 
                output = {'transformed_sources': transformed, 
                      'mapping': load_volume(mapping_file), 
                      'inverse': load_volume(inverse_mapping_file)}
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
    
        target = load_volume(target_images[idx])
        trg_affine = target.affine
        trg_header = target.header
        ntx = target.header.get_data_shape()[X]
        nty = target.header.get_data_shape()[Y]
        ntz = target.header.get_data_shape()[Z]
        rtx = target.header.get_zooms()[X]
        rty = target.header.get_zooms()[Y]
        rtz = target.header.get_zooms()[Z]
    
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
            src_img_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
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
            trg_img_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
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
    src_map_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
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
    trg_map_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                        rootfile=source_images[0],
                                                        suffix='tmp_trgcoord'))
    save_volume(trg_map_file, trg_map)
       
    if mask_zero:
        # create and save temporary masks
        trg_mask_data = (target.get_data()!=0)
        trg_mask = nb.Nifti1Image(trg_mask_data, target.affine, target.header)
        trg_mask_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                            rootfile=source_image,
                                                            suffix='tmp_trgmask'))
        save_volume(trg_mask_file, trg_mask)
        
        src_mask_data = (source.get_data()!=0)
        src_mask = nb.Nifti1Image(src_mask_data, source.affine, source.header)
        src_mask_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                            rootfile=source_image,
                                                            suffix='tmp_srcmask'))
        save_volume(src_mask_file, src_mask)

    # run the main ANTS software: here we directly build the command line call
    reg = 'antsRegistration --collapse-output-transforms 1 --dimensionality 3' \
            +' --initialize-transforms-per-stage 0 --interpolation Linear'
    
     # add a prefix to avoid multiple names?
    prefix = _fname_4saving(file_name=file_name,
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
        if regularization is 'Low': syn_param = [0.2, 1.0, 0.0]
        elif regularization is 'Medium': syn_param = [0.2, 3.0, 0.0]
        elif regularization is 'High': syn_param = [0.2, 4.0, 3.0]
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
        msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
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
        
    print('forward transforms: '+str(forward))    
        
    inverse = []
    linear = []
    for res in results[::-1]:
        if res.endswith('GenericAffine.mat'):
            inverse.append(res)
            linear.append(True)
        elif res.endswith('InverseWarp.nii.gz'):
            inverse.append(res)
            linear.append(False)
     
    print('inverse transforms: '+str(inverse))    
        
    # Transforms the moving image
    transformed = []
    for idx,source in enumerate(sources):
        at = 'antsApplyTransforms --dimensionality 3 --input-image-type 0'
        #at = ApplyTransforms()
        #at.inputs.dimension = 3
        at = at+' --input '+sources[idx].get_filename()
        #at.inputs.input_image = sources[idx].get_filename()
        at = at+' --reference-image '+targets[idx].get_filename()
        #at.inputs.reference_image = targets[idx].get_filename()
        at = at+' --interpolation '+interpolation
        #at.inputs.interpolation = interpolation
        for idx,transform in enumerate(forward):
            if flag[idx]: 
                at = at+' --transform ['+transform+', 1]'
            else:
                at = at+' --transform ['+transform+', 0]'
        #at.inputs.transforms = forward
        #at.inputs.invert_transform_flags = flag
        
        #transformed.append(at.run())
        print(at)
        try:
            subprocess.check_output(at, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            msg = 'execution failed (error code '+e.returncode+')\n Output: '+e.output
            raise subprocess.CalledProcessError(msg)

    # Create coordinate mappings
    src_at = ApplyTransforms()
    src_at.inputs.dimension = 3
    src_at.inputs.input_image_type = 3
    src_at.inputs.input_image = src_map.get_filename()
    src_at.inputs.reference_image = target.get_filename()
    src_at.inputs.interpolation = 'Linear'
    src_at.inputs.transforms = forward
    src_at.inputs.invert_transform_flags = flag
    trans_mapping = src_at.run()

    trg_at = ApplyTransforms()
    trg_at.inputs.dimension = 3
    trg_at.inputs.input_image_type = 3
    trg_at.inputs.input_image = trg_map.get_filename()
    trg_at.inputs.reference_image = source.get_filename()
    trg_at.inputs.interpolation = 'Linear'
    trg_at.inputs.transforms = inverse
    trg_at.inputs.invert_transform_flags = linear
    trans_inverse = trg_at.run()

    # pad coordinate mapping outside the image? hopefully not needed...

    # collect outputs and potentially save
    transformed_imgs = []
    for idx,trans in enumerate(transformed):
        transformed_imgs.append(nb.Nifti1Image(nb.load(trans.outputs.output_image).get_data(), 
                                    targets[idx].affine, targets[idx].header))
    mapping_img = nb.Nifti1Image(nb.load(trans_mapping.outputs.output_image).get_data(), 
                                    targets[0].affine, targets[0].header)
    inverse_img = nb.Nifti1Image(nb.load(trans_inverse.outputs.output_image).get_data(), 
                                    sources[0].affine, sources[0].header)

    outputs = {'transformed_source': transformed_imgs, 'mapping': mapping_img,
                'inverse': inverse_img}

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
    for trans in transformed:
        if os.path.exists(trans.outputs.output_image): 
            os.remove(trans.outputs.output_image)
    if os.path.exists(trans_mapping.outputs.output_image): 
        os.remove(trans_mapping.outputs.output_image)
    if os.path.exists(trans_inverse.outputs.output_image): 
        os.remove(trans_inverse.outputs.output_image)

    if save_data:
        for idx,source_image in enumerate(source_images):
            save_volume(transformed_source_files[idx], transformed_imgs[idx])
        save_volume(mapping_file, mapping_img)
        save_volume(inverse_mapping_file, inverse_img)

    return outputs
