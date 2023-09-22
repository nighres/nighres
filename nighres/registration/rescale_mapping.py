# basic dependencies
import os
import sys

# main dependencies: numpy, nibabel
import numpy
import nibabel
import math

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

def rescale_mapping(source_image=None,
            scaling_factor=1.0,
            save_data=False, overwrite=False, output_dir=None,
            file_name=None):
    """ Rescale

    Generate a pair of coordinate mappings for image rescaling

    Parameters
    ----------
    source_image: niimg
        Image to rescale
    scaling_factor: float
        Factor to use for rescaling
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

        * mapping (niimg): Coordinate mapping from source to rescaled (_sc-map)
        * inverse (niimg): Inverse coordinate mapping from rescaled to source (_sc-invmap)

    Notes
    ----------
    """

    print('\nRescale mapping')

    # make sure that saving related parameters are correct
    output_dir = _output_dir_4saving(output_dir, source_image) # needed for intermediate results
    if save_data:
        mapping_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='sc-map'))

        inverse_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='sc-invmap'))

        if overwrite is False \
           and os.path.isfile(mapping_file) \
            and os.path.isfile(inverse_file) :
            
            print("skip computation (use existing results)")
            output = {'mapping': mapping_file, 'inverse': inverse_file}
            return output


    # load and get dimensions and resolution from input images
    source = load_volume(source_image)
    affine = source.affine
    header = source.header
    nx = source.header.get_data_shape()[X]
    ny = source.header.get_data_shape()[Y]
    nz = source.header.get_data_shape()[Z]
    rx = source.header.get_zooms()[X]
    ry = source.header.get_zooms()[Y]
    rz = source.header.get_zooms()[Z]
    
    nsx = math.ceil(nx*scaling_factor)
    nsy = math.ceil(ny*scaling_factor)
    nsz = math.ceil(nz*scaling_factor)
    rsx = rx/scaling_factor
    rsy = ry/scaling_factor
    rsz = rz/scaling_factor
    header.set_zooms((rsx,rsy,rsz,))

    # build coordinate mapping matrices and save them to disk
    mapping = numpy.zeros((nsx,nsy,nsz,3))
    for x in range(nsx):
        for y in range(nsy):
            for z in range(nsz):
                mapping[x,y,z,X] = x/scaling_factor
                mapping[x,y,z,Y] = y/scaling_factor
                mapping[x,y,z,Z] = z/scaling_factor
                
    inverse = numpy.zeros((nx,ny,nz,3))
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                inverse[x,y,z,X] = x*scaling_factor
                inverse[x,y,z,Y] = y*scaling_factor
                inverse[x,y,z,Z] = z*scaling_factor

    mapping_img = nibabel.Nifti1Image(mapping, affine, header)
    inverse_img = nibabel.Nifti1Image(inverse, affine, header)

    if save_data:
        save_volume(mapping_file, mapping_img)
        save_volume(inverse_file, inverse_img)
        outputs = {'mapping': mapping_file, 'inverse': inverse_file}
    else:
        outputs = {'mapping': mapping_img, 'inverse': inverse_img}

    return outputs
