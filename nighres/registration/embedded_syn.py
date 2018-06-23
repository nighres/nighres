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

def embedded_syn(source_image, target_image, coarse_iterations=40, 
                    medium_iterations=50, fine_iterations=40,
					run_affine_first=False, cost_function='MutualInformation', 
					interpolation='NearestNeighbor',
                    save_data=False, output_dir=None,
                    file_name=None):
    """ Embedded SyN

    Runs the Symmetric Normalization (SyN) algorithm of ANTs and formats the 
    output deformations into voxel coordinate mappings as used in CBSTools 
    registration and transformation routines.

    Parameters
    ----------
    source_image: niimg
        Image to register
    target_image: niimg
        Reference image to match
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
        Interpolation for the registration (default is 'NearestNeighbor')
    save_data: bool
        Save output data to file (default is False)
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

        * deformed_source (niimg): Deformed source image (_syn_def)
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

    print('\nEmbedded SyN')

    # for external tools: nipype
    try:
        from nipype.interfaces.ants import ANTS
        from nipype.interfaces.ants import ApplyTransforms
    except ImportError:
        print('Error: Nipype and/or ANTS could not be imported, they are required' 
                +'in order to run this module. \n (aborting)')
        return None

    # make sure that saving related parameters are correct
    output_dir = _output_dir_4saving(output_dir, source_image) # needed for intermediate results
    if save_data:
        deformed_source_file = _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='syn_def')

        mapping_file = _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='syn_map')

        inverse_mapping_file = _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='syn_invmap')

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
    for x in xrange(nsx):
        for y in xrange(nsy):
            for z in xrange(nsz):
                src_coord[x,y,z,X] = x
                src_coord[x,y,z,Y] = y
                src_coord[x,y,z,Z] = z
    src_map = nb.Nifti1Image(src_coord, source.affine, source.header)
    src_map_file = _fname_4saving(file_name=file_name,
                            rootfile=source_image,
                            suffix='tmp_srccoord')
    save_volume(os.path.join(output_dir, src_map_file), src_map)
    for x in xrange(ntx):
        for y in xrange(nty):
            for z in xrange(ntz):
                trg_coord[x,y,z,X] = x
                trg_coord[x,y,z,Y] = y
                trg_coord[x,y,z,Z] = z
    trg_map = nb.Nifti1Image(trg_coord, target.affine, target.header)
    trg_map_file = _fname_4saving(file_name=file_name,
                            rootfile=source_image,
                            suffix='tmp_trgcoord')
    save_volume(os.path.join(output_dir, trg_map_file), trg_map)
    
    # run the main ANTS software
    ants = ANTS()
    ants.inputs.dimension = 3
    
     # add a prefix to avoid multiple names?
    prefix = _fname_4saving(file_name=file_name,
                            rootfile=source_image,
                            suffix='tmp_syn')
    prefix = os.path.basename(prefix)
    prefix = prefix.split(".")[0]
    ants.inputs.output_transform_prefix = prefix
    if (cost_function=='CrossCorrelation'): 
        ants.inputs.metric = ['CC']
        ants.inputs.metric_weight = [1.0]
        ants.inputs.radius = [5]
    else :
        ants.inputs.metric = ['MI']
        ants.inputs.metric_weight = [1.0]
        ants.inputs.radius = [64]
    ants.inputs.fixed_image = [target_image.get_filename()]
    ants.inputs.moving_image = [source_image.get_filename()]
    ants.inputs.transformation_model = 'SyN'
    ants.inputs.gradient_step_length = 0.25
    ants.inputs.number_of_iterations = [coarse_iterations, medium_iterations, 
                                            fine_iterations]
    #ants.inputs.use_histogram_matching = True
    ants.inputs.mi_option = [32, 16000]
    ants.inputs.regularization = 'Gauss'
    ants.inputs.regularization_gradient_field_sigma = 3
    ants.inputs.regularization_deformation_field_sigma = 1
    ants.inputs.number_of_affine_iterations = [10000,10000,10000,10000,10000]
    result = ants.run()

    # Transforms the moving image
    at = ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = source_image.get_filename()
    at.inputs.reference_image = target_image.get_filename()
    at.inputs.interpolation = interpolation
    at.inputs.transforms = [result.outputs.warp_transform,
                            result.outputs.affine_transform]
    at.inputs.invert_transform_flags = [False, False]
    deformed = at.run()

    # Create coordinate mappings
    src_at = ApplyTransforms()
    src_at.inputs.dimension = 3
    src_at.inputs.input_image_type = 3
    src_at.inputs.input_image = src_map.get_filename()
    src_at.inputs.reference_image = target_image.get_filename()
    src_at.inputs.interpolation = 'Linear'
    src_at.inputs.transforms = [result.outputs.warp_transform,
                                result.outputs.affine_transform]
    src_at.inputs.invert_transform_flags = [False, False]
    mapping = src_at.run()

    trg_at = ApplyTransforms()
    trg_at.inputs.dimension = 3
    trg_at.inputs.input_image_type = 3
    trg_at.inputs.input_image = trg_map.get_filename()
    trg_at.inputs.reference_image = source_image.get_filename()
    trg_at.inputs.interpolation = 'Linear'
    trg_at.inputs.transforms = [result.outputs.affine_transform,
                                result.outputs.inverse_warp_transform]
    trg_at.inputs.invert_transform_flags = [True, False]
    inverse = trg_at.run()

    # pad coordinate mapping outside the image? hopefully not needed...

    # collect outputs and potentially save
    deformed_img = nb.Nifti1Image(nb.load(deformed.outputs.output_image).get_data(), 
                                    target.affine, target.header)
    mapping_img = nb.Nifti1Image(nb.load(mapping.outputs.output_image).get_data(), 
                                    target.affine, target.header)
    inverse_img = nb.Nifti1Image(nb.load(inverse.outputs.output_image).get_data(), 
                                    source.affine, source.header)

    outputs = {'deformed_source': deformed_img, 'mapping': mapping_img,
                'inverse': inverse_img}

    # clean-up intermediate files
    #os.remove(src_map_file)
    #os.remove(trg_map_file)
    os.remove(result.outputs.affine_transform)
    os.remove(result.outputs.warp_transform)
    os.remove(result.outputs.inverse_warp_transform)
    os.remove(deformed.outputs.output_image)
    os.remove(mapping.outputs.output_image)
    os.remove(inverse.outputs.output_image)

    if save_data:
        save_volume(os.path.join(output_dir, deformed_source_file), deformed_img)
        save_volume(os.path.join(output_dir, mapping_file), mapping_img)
        save_volume(os.path.join(output_dir, inverse_mapping_file), inverse_img)

    return outputs
