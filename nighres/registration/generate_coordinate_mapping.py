# basic dependencies
import os
import sys

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

def generate_coordinate_mapping(reference_image, 
                    source_image=None,
                    transform_matrix=None,
                    invert_matrix=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Generate coordiante mapping

    Generate a coordinate mapping for the image(s) and linear transformation 
    as used in CBSTools registration and transformation routines.

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
    convergence: float
        Threshold for convergence, can make the algorithm very slow (default is convergence)
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

        * transformed_source (niimg): Deformed source image (_ants_def)
        * mapping (niimg): Coordinate mapping from source to target (_ants_map)
        * inverse (niimg): Inverse coordinate mapping from target to source (_ants_invmap) 

    Notes
    ----------
    Port of the CBSTools Java module by Pierre-Louis Bazin. Currently the 
    transformation amtrix follows the MIPAV conventions.
    """

    print('\nGenerate coordinate mapping')

    # make sure that saving related parameters are correct
    output_dir = _output_dir_4saving(output_dir, source_image) # needed for intermediate results
    if save_data:
        mapping_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=source_image,
                                   suffix='coord-map'))

        if overwrite is False \
            and os.path.isfile(mapping_file) :
            
            print("skip computation (use existing results)")
            output = {'result': load_volume(mapping_file)}
            return output


    # load and get dimensions and resolution from input images
    reference = load_volume(reference_image)
    ref_affine = reference.affine
    ref_header = reference.header
    nx = reference.header.get_data_shape()[X]
    ny = reference.header.get_data_shape()[Y]
    nz = reference.header.get_data_shape()[Z]
    rtx = reference.header.get_zooms()[X]
    rty = reference.header.get_zooms()[Y]
    rtz = reference.header.get_zooms()[Z]

    rsx = rtx
    rsy = rty
    rsz = rtz
    if source is not None:
        source = load_volume(source_image)
        rsx = source.header.get_zooms()[X]
        rsy = source.header.get_zooms()[Y]
        rsz = source.header.get_zooms()[Z]

    if transform_matrix is not None:
        with open(transform_matrix, 'r+') as f:
            # assuming the MIPAV file type here (others would need modification)
            f.readline()
            f.readline()
            str_text = f.readline()+f.readline()+f.readline()+f.readline()
            str_list = str_text.split('\n')
            str_array = []
            for sl in str_list:
                str_array.append(sl.split())
            transform = numpy.array(str_array,dtype='float')
    else:
        transform = numpy.eye(4,4)
        
    if not invert_matrix:
        transform.invert()
        
    # build coordinate mapping matrices and save them to disk
    coord = np.zeros((nx,ny,nz,3))
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                coord[x,y,z,X] = (transform[X,X]*x*rtx \
                                + transform[X,Y]*y*rty \
                                + transform[X,Z]*z*rtz \
                                + transform[X,T])/rsx
                coord[x,y,z,Y] = (transform[Y,X]*x*rtx \
                                + transform[Y,Y]*y*rty \
                                + transform[Y,Z]*z*rtz \
                                + transform[Y,T])/rsy
                coord[x,y,z,Z] = (transform[Z,X]*x*rtx \
                                + transform[Z,Y]*y*rty \
                                + transform[Z,Z]*z*rtz \
                                + transform[Z,T])/rsz
                
    mapping_img = nb.Nifti1Image(src_coord, source.affine, source.header)
    src_map_file = os.path.join(output_dir, _fname_4saving(file_name=file_name,
                                                        rootfile=source_image,
                                                        suffix='tmp_srccoord'))
    if save:
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

    # collect outputs and potentially save
    mapping_img = nb.Nifti1Image(nb.load(mapping.outputs.output_image).get_data(), 
                                    target.affine, target.header)

    outputs = {'result': mapping_img}

    return outputs

