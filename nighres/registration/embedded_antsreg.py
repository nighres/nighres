# basic dependencies
import os
import sys

# main dependencies: numpy, nibabel
import numpy as np
import nibabel as nb

# cbstools and nighres functions
import cbstools
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
                    run_syn=True,
                    coarse_iterations=40, 
                    medium_iterations=50, fine_iterations=40,
					cost_function='MutualInformation', 
					interpolation='NearestNeighbor',
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
    run_syn: bool
        Whether or not to run a SyN registration (default is True)
    coarse_iterations: float
        Number of iterations at the coarse level (default is 40)
    medium_iterations: float
        Number of iterations at the medium level (default is 50)
    fine_iterations: float
        Number of iterations at the fine level (default is 40)
    run_affine_first: bool
        Runs a step of affine registration before the non-linear step (default is False)
    cost_function: {'CrossCorrelation', 'MutualInformation'}
        Cost function for the registration (default is 'MutualInformation')
    interpolation: {'NearestNeighbor', 'Linear'}
        Interpolation for the registration result (default is 'NearestNeighbor')
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

        * transformed_source (niimg): Deformed source image (_syn_def)
        * mapping (niimg): Coordinate mapping from source to target (_syn_map)
        * inverse (niimg): Inverse coordinate mapping from target to source (_syn_invmap) 

    Notes
    ----------
    Port of the CBSTools Java module by Pierre-Louis Bazin. The main algorithm
    is part of the ANTs software by Brian Avants and colleagues [1]_. The interfacing
    with ANTs is performed through Nipype [2]_.

    References
    ----------
    .. [1] Avants et al (2008), Symmetric diffeomorphic 
       image registration with cross-correlation: evaluating automated labeling 
       of elderly and neurodegenerative brain, Med Image Anal. 12(1):26-41
    .. [2] Gorgolewski et al (2011) Nipype: a flexible, lightweight and extensible 
       neuroimaging data processing framework in python. Front Neuroinform 5. 
       doi:10.3389/fninf.2011.00013
    """

    print('\nEmbedded ANTs Registration')

    # for external tools: nipype
    try:
        from nipype.interfaces.ants import Registration
        from nipype.interfaces.ants import ApplyTransforms
    except ImportError:
        print('Error: Nipype and/or ANTS could not be imported, they are required' 
                +'in order to run this module. \n (aborting)')
        return None

    # make sure that saving related parameters are correct
    output_dir = _output_dir_4saving(output_dir, source_image) # needed for intermediate results
    if save_data:
        transformed_source_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='syn-def'))

        mapping_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='syn-map'))

        inverse_mapping_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='syn-invmap'))
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
    
    # run the main ANTS software
    reg = Registration()
    reg.inputs.dimension = 3
    
     # add a prefix to avoid multiple names?
    prefix = _fname_4saving(file_name=file_name,
                            rootfile=source_image,
                            suffix='tmp_syn')
    prefix = os.path.basename(prefix)
    prefix = prefix.split(".")[0]
    reg.inputs.output_transform_prefix = prefix
    reg.inputs.fixed_image = [target.get_filename()]
    reg.inputs.moving_image = [source.get_filename()]
    
    if run_rigid is True and run_syn is True:
        reg.inputs.transforms = ['Rigid','SyN']
        reg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
        reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations, 
                                            rigid_iterations],   
                                           [coarse_iterations, medium_iterations, 
                                            fine_iterations]]
        if (cost_function=='CrossCorrelation'): 
            reg.inputs.metric = ['CC', 'CC']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [5, 5]
        else :
            reg.inputs.metric = ['MI', 'MI']
            reg.inputs.metric_weight = [1.0, 1.0]
            reg.inputs.radius_or_number_of_bins = [64, 64]
        reg.inputs.shrink_factors = [[3, 2, 1]] + [[4, 2, 1]]   
        reg.inputs.smoothing_sigmas = [[4, 2, 1]] + [[1, 0.5, 0]]

    elif run_rigid is True and run_syn is False:
        reg.inputs.transforms = ['Rigid']
        reg.inputs.transform_parameters = [(2.0,)]
        reg.inputs.number_of_iterations = [[rigid_iterations, rigid_iterations,
                                            rigid_iterations]]
        if (cost_function=='CrossCorrelation'): 
            reg.inputs.metric = ['CC']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [5]
        else :
            reg.inputs.metric = ['MI']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [64]
        reg.inputs.shrink_factors = [[3, 2, 1]] 
        reg.inputs.smoothing_sigmas = [[4, 2, 1]] 

    elif run_rigid is False and run_syn is True:
        reg.inputs.transforms = ['SyN']
        reg.inputs.transform_parameters = [(0.25, 3.0, 0.0)]
        reg.inputs.number_of_iterations = [[coarse_iterations, medium_iterations, 
                                            fine_iterations]]
        if (cost_function=='CrossCorrelation'): 
            reg.inputs.metric = ['CC']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [5]
        else :
            reg.inputs.metric = ['MI']
            reg.inputs.metric_weight = [1.0]
            reg.inputs.radius_or_number_of_bins = [64]
        reg.inputs.shrink_factors = [[4, 2, 1]]
        reg.inputs.smoothing_sigmas = [[1, 0.5, 0]]

    elif run_rigid is False and run_syn is False:
        reg.inputs.transforms = ['Rigid']
        reg.inputs.transform_parameters = [(2.0,)]
        reg.inputs.number_of_iterations = [[0]]
        reg.inputs.metric = ['CC']
        reg.inputs.metric_weight = [1.0]
        reg.inputs.radius_or_number_of_bins = [5]
        reg.inputs.shrink_factors = [[1]] 
        reg.inputs.smoothing_sigmas = [[1]]

        
        
    #print(reg.cmdline)
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
    for name in result.outputs.forward_transforms: 
        if os.path.exists(name): os.remove(name)
    for name in result.outputs.reverse_transforms: 
        if os.path.exists(name): os.remove(name)
    os.remove(mapping.outputs.output_image)
    os.remove(inverse.outputs.output_image)

    if save_data:
        save_volume(transformed_source_file, transformed_img)
        save_volume(mapping_file, mapping_img)
        save_volume(inverse_mapping_file, inverse_img)

    return outputs
