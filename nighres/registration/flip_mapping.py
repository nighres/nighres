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

def flip_mapping(source_image, flip,
            save_data=False, overwrite=False, output_dir=None,
            file_name=None):
    """ Flip

    Generate a pair of coordinate mappings for flipping images along one of the axes

    Parameters
    ----------
    source_image: niimg
        Image to rescale
    flip: str
        Axis to use for flipping ('X', 'Y' or 'Z')
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

        * mapping (niimg): Coordinate mapping from source to flipped (_sc-map)
        * inverse (niimg): Inverse coordinate mapping from flipped to source (which is the same 
                           transform, we link to it for consistency with other mappings)

    Notes
    ----------
    """

    print('\nFlip mapping')

    # make sure that saving related parameters are correct
    output_dir = _output_dir_4saving(output_dir, source_image) # needed for intermediate results
    if save_data:
        mapping_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='fl-map'))

        if overwrite is False \
           and os.path.isfile(mapping_file) :
            
            print("skip computation (use existing results)")
            output = {'mapping': mapping_file, 'inverse': mapping_file}
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
    
    # recode the axis
    direction = -1
    if flip=='X': 
        direction = X
    elif flip=='Y':
        direction = Y
    elif flip=='Z':
        direction = Z
        
    if direction==-1:
        print('incorrect axis definition')
        return
    
    # build coordinate mapping matrices and save them to disk
    mapping = numpy.zeros((nx,ny,nz,3))
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if direction==X:
                    mapping[x,y,z,X] = nx-1-x
                else: 
                    mapping[x,y,z,X] = x
                if direction==Y:
                    mapping[x,y,z,Y] = ny-1-y
                else:
                    mapping[x,y,z,Y] = y
                if direction==Z:
                    mapping[x,y,z,Z] = nz-1-z
                else:
                    mapping[x,y,z,Z] = z
                
    mapping_img = nibabel.Nifti1Image(mapping, affine, header)

    if save_data:
        save_volume(mapping_file, mapping_img)
        outputs = {'mapping': mapping_file, 'inverse': mapping_file}
    else:
        outputs = {'mapping': mapping_img, 'inverse': mapping_img}

    return outputs
