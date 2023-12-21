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

def crop_mapping(source_image=None,
            boundary=2,
            save_data=False, overwrite=False, output_dir=None,
            file_name=None):
    """ Crop mapping

    Generate a pair of coordinate mappings for image cropping

    Parameters
    ----------
    source_image: niimg
        Image to crop
    boundary: int
        Added voxels at the image boundary
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

    print('\nCrop mapping')

    # make sure that saving related parameters are correct
    output_dir = _output_dir_4saving(output_dir, source_image) # needed for intermediate results
    if save_data:
        mapping_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='cr-map'))

        inverse_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='cr-invmap'))

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
    data = source.get_fdata()
    
    xmin = numpy.min(numpy.nonzero(data)[0])
    xmax = numpy.max(numpy.nonzero(data)[0])
    ymin = numpy.min(numpy.nonzero(data)[1])
    ymax = numpy.max(numpy.nonzero(data)[1])
    zmin = numpy.min(numpy.nonzero(data)[2])
    zmax = numpy.max(numpy.nonzero(data)[2])
    
    nsx = xmax-xmin+2*boundary
    nsy = ymax-ymin+2*boundary
    nsz = zmax-zmin+2*boundary

    # build coordinate mapping matrices and save them to disk
    mapping = numpy.zeros((nsx,nsy,nsz,3))
    inverse = numpy.zeros((nx,ny,nz,3))
    for x in range(nsx):
        for y in range(nsy):
            for z in range(nsz):
                mapping[x,y,z,X] = x+xmin-boundary
                mapping[x,y,z,Y] = y+ymin-boundary
                mapping[x,y,z,Z] = z+zmin-boundary

                if x+xmin-boundary>=0 and y+ymin-boundary>=0 and z+zmin-boundary>=0 \
                    and  x+xmin-boundary<nx and y+ymin-boundary<ny and z+zmin-boundary<nz:
                    inverse[x+xmin-boundary,y+ymin-boundary,z+zmin-boundary,X] = x
                    inverse[x+xmin-boundary,y+ymin-boundary,z+zmin-boundary,Y] = y
                    inverse[x+xmin-boundary,y+ymin-boundary,z+zmin-boundary,Z] = z

    mapping_img = nibabel.Nifti1Image(mapping, affine, header)
    inverse_img = nibabel.Nifti1Image(inverse, affine, header)

    if save_data:
        save_volume(mapping_file, mapping_img)
        save_volume(inverse_file, inverse_img)
        outputs = {'mapping': mapping_file, 'inverse': inverse_file}
    else:
        outputs = {'mapping': mapping_img, 'inverse': inverse_img}

    return outputs
