import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
    _check_topology_lut_dir, _check_available_memory

def filter_ridge_structures(input_image,
                            structure_intensity='bright',
                            output_type='probability',
                            use_strict_min_max_filter=True,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):

    """ Filter Ridge Structures

    Uses an image filter to make a probabilistic image of ridge
    structures.


    Parameters
    ----------
    input_image: niimg
        Image containing structure-of-interest
    structure_intensity: {'bright', 'dark', 'both}
        Image intensity of structure-of-interest'
    output_type: {'probability','intensity'}
        Whether the image should be normalized to reflect probabilities
    use_strict_min_max_filter: bool, optional
        Choose between the more specific recursive ridge filter or a more
        sensitive bidirectional filter (default is True)
    save_data: bool, optional
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

        * ridge_structure_image: Image that reflects the presensence of ridges
          in the image (_rdg-img)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_image)

        ridge_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=input_image,
                                       suffix='rdg-img', ))
        if overwrite is False \
            and os.path.isfile(ridge_file) :

            print("skip computation (use existing results)")
            output = {'result': ridge_file}
            return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    filter_ridge = nighresjava.FilterRidgeStructures()

    # set parameters
    filter_ridge.setStructureIntensity(structure_intensity)
    filter_ridge.setOutputType(output_type)
    filter_ridge.setUseStrictMinMaxFilter(use_strict_min_max_filter)


    # load images and set dimensions and resolution
    input_image = load_volume(input_image)
    data = input_image.get_data()
    affine = input_image.affine
    header = input_image.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = input_image.shape


    filter_ridge.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    filter_ridge.setResolutions(resolution[0], resolution[1], resolution[2])

    data = load_volume(input_image).get_data()
    filter_ridge.setInputImage(nighresjava.JArray('float')(
                               (data.flatten('F')).astype(float)))


    # execute
    try:
        filter_ridge.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
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

    ridge_structure_image = nb.Nifti1Image(ridge_structure_image_data, affine,
                                           header)
    if save_data:
        save_volume(ridge_file, ridge_structure_image)
        outputs = {'result': ridge_file}
    else:
        outputs = {'result': ridge_structure_image}
    return outputs
