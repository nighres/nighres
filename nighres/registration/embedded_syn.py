# basic dependencies
import os
import sys

# main dependencies: numpy, nibabel
import numpy as np
import nibabel as nb

# for external tools: nipype
from nipype.interfaces.ants import ANTS
from nipype.interfaces.ants import WarpImageMultiTransform
from nipype.interfaces.ants import WarpTimeSeriesImageMultiTransform

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

def embedded_syn(source_image, target_image, coarse_iterations=40, medium_iterations=50, fine_iterations=40,
					run_affine_first=False, cost_function='Mutual Information', interpolation='Nearest Neighbor',
                    save_data=False, output_dir=None,
                    file_name=None):
    """ Embedded SyN

    Runs the Symmetric Normalization (SyN) algorithm of ANTs and formats the output deformations
    into voxel coordinate mappings as used in CBSTools registration and transformation routines.

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
    cost_function: {'Cross Correlation', 'Mutual Information'}
        Cost function for the registration (default is 'Mutual Information')
    interpolation: {'Nearest Neighbor', 'Linear'}
        Cost function for the registration (default is 'Nearest Neighbor')
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

        * deformed_source (niimg): Deformed source image (_esyn_def)
        * mapping (niimg): Coordinate mapping from source to target (_esyn_map)
        * inverse (niimg): Inverse coordinate mapping from target to source (_esyn_invmap) 

    Notes
    ----------
    Port of the CBSTools Java module by Pierre-Louis Bazin. The main algorithm
    is part of the ANTs software by Brian Avants and colleagues [1]_

    References
    ----------
    .. [1] Avants BB, Epstein CL, Grossman M, Gee JC, Symmetric diffeomorphic 
       image registration with cross-correlation: evaluating automated labeling 
       of elderly and neurodegenerative brain, Med Image Anal. 2008 Feb;12(1):26-41
    """

    print('\nEmbedded SyN')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, second_inversion)

        deformed_source_file = _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='esyn_def')

        mapping_file = _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='esyn_map')

        inverse_mapping_file = _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='esyn_invmap')

     # load and get dimensions and resolution from input images
    source = load_volume(source_image)
	src_affine = source.affine
	src_header = source.header
	nsx = source.header.getShape()[X]
	nsy = source.header.getShape()[Y]
	nsz = source.header.getShape()[Z]
	rsx = source.header.getResolutions()[X]
	rsy = source.header.getResolutions()[Y]
	rsz = source.header.getResolutions()[Z]

    target = load_volume(target_image)
	trg_affine = target.affine
	trg_header = target.header
	ntx = target.header.getShape()[X]
	nty = target.header.getShape()[Y]
	ntz = target.header.getShape()[Z]
	rtx = target.header.getResolutions()[X]
	rty = target.header.getResolutions()[Y]
	rtz = target.header.getResolutions()[Z]

	# build coordinate mapping matrices
	src_coord = np.zeros((nsx,nsy,nsz,3))
	trg_coord = np.zeros((ntx,nty,ntz,3))
	for x in xrange(nsx):
		for y in xrange(nsy):
			for z in xrange(nsz):
				src_coord[x,y,z,X] = x
				src_coord[x,y,z,Y] = y
				src_coord[x,y,z,Z] = z
	for x in xrange(ntx):
		for y in xrange(nty):
			for z in xrange(ntz):
				trg_coord[x,y,z,X] = x
				trg_coord[x,y,z,Y] = y
				trg_coord[x,y,z,Z] = z

	# run the main ANTS software
   	ants = ANTS()
    ants.inputs.dimension = 3
    #ants.inputs.output_transform_prefix = 'MY'
    if (cost_function=='Cross Correlation'): 
    	ants.inputs.metric = ['CC']
    	ants.inputs.metric_weight = [1.0]
    	ants.inputs.radius = [5]
    else :
    	ants.inputs.metric = ['MI']
    	ants.inputs.metric_weight = [1.0]
    	ants.inputs.radius = [64]
    ants.inputs.fixed_image = [target_image]
    ants.inputs.moving_image = [source_image]
    ants.inputs.transformation_model = 'SyN'
    ants.inputs.gradient_step_length = 0.25
    ants.inputs.number_of_iterations = [coarse_iterations, medium_iterations, fine_iterations]
    #ants.inputs.use_histogram_matching = True
    ants.inputs.mi_option = [32, 16000]
    ants.inputs.regularization = 'Gauss'
    ants.inputs.regularization_gradient_field_sigma = 3
    ants.inputs.regularization_deformation_field_sigma = 1
    ants.inputs.number_of_affine_iterations = [10000,10000,10000,10000,10000]
    ants.cmdline # doctest: +ALLOW_UNICODE
    # main command? ants.run()

	# Transforms the moving image
	wimt = WarpImageMultiTransform()
    wimt.inputs.input_image = source_image
    wimt.inputs.reference_image = target_image
    if (interpolation=='Nearest Neighbor') : wimt.inputs.interpolation = 'NN'
    else : wimt.inputs.interpolation = 'Linear'
    wimt.inputs.transformation_series = ['synWarp.nii','synAffine.txt']
    wtsimt.cmdline # doctest: +ALLOW_UNICODE

	src_wimt = WarpTimeSeriesImageMultiTransform()
    src_wimt.inputs.input_image = src_coord
    src_wimt.inputs.reference_image = target_image
    src_wimt.inputs.transformation_series = ['synWarp.nii','synAffine.txt']
    src_wimt.cmdline # doctest: +ALLOW_UNICODE

	trg_wimt = WarpTimeSeriesImageMultiTransform()
    trg_wimt.inputs.input_image = trg_coord
    trg_wimt.inputs.reference_image = source_image
    trg_wimt.inputs.transformation_series = ['synAffine.txt','synInverseWarp.nii']
    trg_wimt.inputs.invert_affine = True
    trg_wimt.cmdline # doctest: +ALLOW_UNICODE

    # collect outputs and potentially save
	deformed_img = wimt.getWarpedImage()
	mapping_img = src_wimt.getWarpedImage()
	inverse_img = trg_wimt.getWarpedImage()

    outputs = {'deformed_source': deformed_img, 'mapping': mapping_img, 'inverse': inverse_img}

    if save_data:
        save_volume(os.path.join(output_dir, deformed_source_file), deformed_img)
        save_volume(os.path.join(output_dir, mapping_file), mapping_img)
        save_volume(os.path.join(output_dir, inverse_mapping_file), inverse_img)

    return outputs
