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
    _check_topology_lut_dir, _check_atlas_file, _check_available_memory


def levelset_thickness(input_image,
                    shape_image_type='signed_distance',
                   save_data=False, 
                   overwrite=False, 
                   output_dir=None,
                   file_name=None):

    """ Levelset Thickness
    
    Using a medial axis representation, derive a thickness map for a levelset surface


    Parameters
    ----------
    input_image: niimg
        Image containing structure-of-interest
    shape_image_type: str
        shape of the input image: either 'signed_distance' or 'probability_map'.
    save_data: bool, optional
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

        * thickness (niimg): Estimated thickness map (_lth-map)
        * axis (niimg): Medial axis extracted (_lth-ax)
        * dist (niimg): Medial axis distance (_lth-dist)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_image)

        thickness_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='_lth-map', ))

        axis_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='_lth-ax'))    

        dist_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='_lth-dist'))  

        if overwrite is False \
            and os.path.isfile(thickness_file) \
            and os.path.isfile(axis_file) \
            and os.path.isfile(dist_file) :
                output = {'thickness': thickness_file,
                          'axis':axis_file,
                          'dist':dist_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    algorithm = nighresjava.LevelsetThickness()

    # set parameters
    algorithm.setShapeImageType(shape_image_type)


    # load images and set dimensions and resolution
    input_image = load_volume(input_image)
    data = input_image.get_data()
    affine = input_image.get_affine()
    header = input_image.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = input_image.shape


    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])

    data = load_volume(input_image).get_data()
    algorithm.setShapeImage(nighresjava.JArray('float')(
                               (data.flatten('F')).astype(float)))

    # execute
    try:
        algorithm.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # Collect output
    axis_data = np.reshape(np.array(
                                    algorithm.getMedialAxisImage(),
                                    dtype=np.int8), dimensions, 'F')
    dist_data = np.reshape(np.array(
                                    algorithm.getMedialDistanceImage(),
                                    dtype=np.int8), dimensions, 'F')

    thick_data = np.reshape(np.array(
                                    algorithm.geThicknessImage(),
                                    dtype=np.int8), dimensions, 'F')


    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(medialImage_data)
    header['cal_max'] = np.nanmax(medialImage_data)
    axis_img = nb.Nifti1Image(axis_data, affine, header)

    header['cal_min'] = np.nanmin(skelImage_data)
    header['cal_max'] = np.nanmax(skelImage_data)
    dist_img = nb.Nifti1Image(dist_data, affine, header)

    header['cal_min'] = np.nanmin(skelImage_data)
    header['cal_max'] = np.nanmax(skelImage_data)
    thick_img = nb.Nifti1Image(thick_data, affine, header)

    if save_data:
        save_volume(axis_file, axis_img)
        save_volume(dist_file), dist_img)
        save_volume(thick_file), thick_img)

        return {'thickness': thick_file, 
                'axis': axis_file,
                'dist': dist_file}
    else:
        return {'thickness': thick_img, 'axis': axis_img, 'dist': dist_img}


