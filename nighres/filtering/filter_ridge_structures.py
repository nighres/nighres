import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
    _check_topology_lut_dir, _check_atlas_file

def filter_ridge_structures(input_image,
                            structure_intensity='bright',
                            output_type='probability',
                            use_strict_min_max_filter=True,
                            save_data=False, output_dir=None,
                            file_name=None):

    """ Filter Ridge Structures
    
    Uses an image filter to make a probabilistic image of ridge
    structures.


    Parameters
    ----------
    input_image: niimg
        Image containing structure-of-interest
    structure_intensity: str
        Image intensity of structure-of-interest 'bright', 'dark', or 'both'.
    output_type: str
        Whether the image should be normalized to reflect probabilities ('probability'
        or 'intensity'
    use_strict_min_max_filter: bool, optional (defaulti s True)
        Choose between the more specific recursive ridge filter or a more sensitive bidirectional filter
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

        * ridge_structure_image: Image that reflects the presensence of ridges
          in the image

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_image)

        ridge_file = _fname_4saving(file_name=file_name,
                                       rootfile=input_image,
                                       suffix='rdg', )
    outputs = {}

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass
    # create algorithm instance
    filter_ridge = cbstools.FilterRidgeStructures()

    # set parameters
    filter_ridge.setStructureIntensity(structure_intensity)
    filter_ridge.setOutputType(output_type)
    filter_ridge.setUseStrictMinMaxFilter(use_strict_min_max_filter)


    # load images and set dimensions and resolution
    input_image = load_volume(input_image)
    data = input_image.get_data()
    affine = input_image.get_affine()
    header = input_image.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = input_image.shape


    filter_ridge.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    filter_ridge.setResolutions(resolution[0], resolution[1], resolution[2])

    data = load_volume(input_image).get_data()
    filter_ridge.setInputImage(cbstools.JArray('float')(
                               (data.flatten('F')).astype(float)))


    # execute
    try:
        filter_ridge.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

    # Collect output
    ridge_structure_image_data = np.reshape(np.array(
                                    filter_ridge.getRidgeStructureImage(),
                                    dtype=np.float32), dimensions, 'F')

    if output_type == 'probability':
        header['cal_min'] = 0.0
        header['cal_max'] = 1.0
    else:
        header['cal_min'] = np.nanmin(ridge_structure_image_data)
        header['cal_max'] = np.nanmax(ridge_structure_image_data)

    ridge_structure_image = nb.Nifti1Image(ridge_structure_image_data, affine, header)
    outputs['ridge_structure_image'] = ridge_structure_image


    if save_data:
        save_volume(os.path.join(output_dir, ridge_file), ridge_structure_image)

    return outputs



