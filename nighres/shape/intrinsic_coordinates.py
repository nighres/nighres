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
    _check_topology_lut_dir, _check_available_memory


def intrinsic_coordinates(label_image,
                   system_type='centroid_pca',
                   som_size=10,
                   save_data=False, 
                   overwrite=False, 
                   output_dir=None,
                   file_name=None):

    """ Intrinsic Coordinates
    
    Derive an intrinsic coordinate system based on a given parcellation


    Parameters
    ----------
    label_image: niimg
        Image of the object(s) of interest
    system_type: str
        coordinate system derivation: 'centroid_pca', 'weighted_pca', 'voxel_pca', 'weighted_som', 'weighted_som2d'.
    som_size: int
        size of the self-organizing map for the 'weighted_som' (default is 10)
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

        * coordinates (niimg): Coordinate map (_ics-coord)
        * image (niimg): Input image in the new coordinate system (_ics-img)

    Notes
    ----------
    """

    print("\nIntrinsic Coordinates")

    if save_data:
        output_dir = _output_dir_4saving(output_dir, label_image)

        coord_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=label_image,
                                  suffix='ics-coord'))

        img_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=label_image,
                                  suffix='ics-img'))    

        if overwrite is False \
            and os.path.isfile(coord_file) \
            and os.path.isfile(img_file) :
                output = {'coordinates': coord_file,
                          'image': img_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    algorithm = nighresjava.IntrinsicCoordinates()

    # set parameters
    algorithm.setSystemType(system_type)
    algorithm.setSomSize(som_size)

    # load images and set dimensions and resolution
    label_image = load_volume(label_image)
    data = label_image.get_data()
    affine = label_image.get_affine()
    header = label_image.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = label_image.shape
    dimensions4 = (dimensions[0],dimensions[1],dimensions[2],3)


    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])

    data = load_volume(label_image).get_data()
    algorithm.setLabelImage(nighresjava.JArray('int')(
                               (data.flatten('F')).astype(int).tolist()))

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
    coord_data = np.reshape(np.array(
                                    algorithm.getCoordinateImage(),
                                    dtype=np.float32), dimensions4, 'F')
    img_data = np.reshape(np.array(
                                    algorithm.getTransformedImage(),
                                    dtype=np.int32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(coord_data)
    header['cal_max'] = np.nanmax(coord_data)
    coord_img = nb.Nifti1Image(coord_data, affine, header)

    header['cal_min'] = np.nanmin(img_data)
    header['cal_max'] = np.nanmax(img_data)
    img_img = nb.Nifti1Image(img_data, affine, header)

    if save_data:
        save_volume(coord_file, coord_img)
        save_volume(img_file, img_img)
        
        return {'coordinates': coord_file, 
                'image': img_file}
    else:
        return {'coordinates': coord_img, 'image': img_img}


