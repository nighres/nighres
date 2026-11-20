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
    scaling_factor: float or tuple
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
    if save_data:
        output_dir = _output_dir_4saving(output_dir, source_image) # needed for intermediate results
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

    # set the scales
    if isinstance(scaling_factor, tuple) and len(scaling_factor)==3:
        scalingX = scaling_factor[0]
        scalingY = scaling_factor[1]
        scalingZ = scaling_factor[2]
    else:
        scalingX = scaling_factor
        scalingY = scaling_factor
        scalingZ = scaling_factor
        

    # load and get dimensions and resolution from input images
    source = load_volume(source_image)
    affine = source.affine
    srcheader =  source.header
    trgheader = source.header.copy()
    nx = source.header.get_data_shape()[X]
    ny = source.header.get_data_shape()[Y]
    nz = source.header.get_data_shape()[Z]
    rx = source.header.get_zooms()[X]
    ry = source.header.get_zooms()[Y]
    rz = source.header.get_zooms()[Z]
    
    nsx = math.ceil(nx*scalingX)
    nsy = math.ceil(ny*scalingY)
    nsz = math.ceil(nz*scalingZ)
    rsx = rx/scalingX
    rsy = ry/scalingY
    rsz = rz/scalingZ
    if len(source.header.get_zooms())==3:
        trgheader.set_zooms((rsx,rsy,rsz))
    elif len(source.header.get_zooms())==4:
        trgheader.set_zooms((rsx,rsy,rsz,source.header.get_zooms()[3]))

    # build coordinate mapping matrices and save them to disk
    
    # slow version
    #mapping = numpy.zeros((nsx,nsy,nsz,3))
    #for x in range(nsx):
    #    for y in range(nsy):
    #        for z in range(nsz):
    #            mapping[x,y,z,X] = x/scalingX
    #            mapping[x,y,z,Y] = y/scalingY
    #            mapping[x,y,z,Z] = z/scalingZ
    #            
    #inverse = numpy.zeros((nx,ny,nz,3))
    #for x in range(nx):
    #    for y in range(ny):
    #        for z in range(nz):
    #            inverse[x,y,z,X] = x*scalingX
    #            inverse[x,y,z,Y] = y*scalingY
    #            inverse[x,y,z,Z] = z*scalingZ

    # faster alternative?
    mapping = numpy.ones((nsx,nsy,nsz))
    mapx = numpy.arange(nsx)[:,numpy.newaxis,numpy.newaxis]*mapping/scalingX
    mapy = numpy.arange(nsy)[numpy.newaxis,:,numpy.newaxis]*mapping/scalingY
    mapz = numpy.arange(nsz)[numpy.newaxis,numpy.newaxis,:]*mapping/scalingZ
    mapping = numpy.stack([mapx,mapy,mapz],axis=-1)

    inverse = numpy.ones((nx,ny,nz))
    invx = numpy.arange(nx)[:,numpy.newaxis,numpy.newaxis]*inverse*scalingX
    invy = numpy.arange(ny)[numpy.newaxis,:,numpy.newaxis]*inverse*scalingY
    invz = numpy.arange(nz)[numpy.newaxis,numpy.newaxis,:]*inverse*scalingZ
    inverse = numpy.stack([invx,invy,invz],axis=-1)

    mapping_img = nibabel.Nifti1Image(mapping, affine, trgheader)
    inverse_img = nibabel.Nifti1Image(inverse, affine, srcheader)

    if save_data:
        save_volume(mapping_file, mapping_img)
        save_volume(inverse_file, inverse_img)
        outputs = {'mapping': mapping_file, 'inverse': inverse_file}
    else:
        outputs = {'mapping': mapping_img, 'inverse': inverse_img}

    return outputs


def rescale_mapping_2d(source_image=None,
            scaling_factor=1.0,
            save_data=False, overwrite=False, output_dir=None,
            file_name=None):
    """ Rescale

    Generate a pair of coordinate mappings for image rescaling in 2D

    Parameters
    ----------
    source_image: niimg
        Image to rescale
    scaling_factor: float or tuple
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

    print('\nRescale mapping 2D')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, source_image) # needed for intermediate results
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

    # set the scales
    if isinstance(scaling_factor, tuple) and len(scaling_factor)==2:
        scalingX = scaling_factor[0]
        scalingY = scaling_factor[1]
    else:
        scalingX = scaling_factor
        scalingY = scaling_factor

    # load and get dimensions and resolution from input images
    source = load_volume(source_image)
    affine = source.affine
    srcheader =  source.header
    trgheader = source.header.copy()
    nx = source.header.get_data_shape()[X]
    ny = source.header.get_data_shape()[Y]
    rx = source.header.get_zooms()[X]
    ry = source.header.get_zooms()[Y]
    
    nsx = math.ceil(nx*scalingX)
    nsy = math.ceil(ny*scalingY)
    rsx = rx/scaling_factor
    rsy = ry/scaling_factor
    if len(source.header.get_zooms())==2:
        trgheader.set_zooms((rsx,rsy))
    elif len(source.header.get_zooms())==3:
        trgheader.set_zooms((rsx,rsy,source.header.get_zooms()[3]))

    # build coordinate mapping matrices and save them to disk
    
    # slow version
    #mmapping = numpy.zeros((nsx,nsy,2))
    #for x in range(nsx):
    #    for y in range(nsy):
    #            mapping[x,y,X] = x/scalingX
    #            mapping[x,y,Y] = y/scalingY
    #            
    #inverse = numpy.zeros((nx,ny,2))
    #for x in range(nx):
    #    for y in range(ny):
    #        inverse[x,y,X] = x*scalingX
    #        inverse[x,y,Y] = y*scalingY

    # faster alternative?
    mapping = numpy.ones((nsx,nsy))
    mapx = numpy.arange(nsx)[:,numpy.newaxis]*mapping/scalingX
    mapy = numpy.arange(nsy)[numpy.newaxis,:]*mapping/scalingY
    mapping = numpy.stack([mapx,mapy],axis=-1)

    inverse = numpy.ones((nx,ny))
    invx = numpy.arange(nx)[:,numpy.newaxis,numpy.newaxis]*inverse*scalingX
    invy = numpy.arange(ny)[numpy.newaxis,:,numpy.newaxis]*inverse*scalingY
    inverse = numpy.stack([invx,invy],axis=-1)
    
    mapping_img = nibabel.Nifti1Image(mapping, affine, trgheader)
    inverse_img = nibabel.Nifti1Image(inverse, affine, srcheader)

    if save_data:
        save_volume(mapping_file, mapping_img)
        save_volume(inverse_file, inverse_img)
        outputs = {'mapping': mapping_file, 'inverse': inverse_file}
    else:
        outputs = {'mapping': mapping_img, 'inverse': inverse_img}

    return outputs
