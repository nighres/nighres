import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def stack_intensity_regularisation(image, cutoff=50, mask=None,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):
    """ Stack intensity regularisation

    Estimates an image-to-image linear intensity scaling for a stack of 2D images

    Parameters
    ----------
    image: niimg
        Input 2D images, stacked in the Z dimension
    cutoff: float, optional 
        Range of image differences to keep (default is middle 50%)
    mask: niimg
        Input mask or probability image of the data to use (optional)
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

        * result (niimg): The intensity regularised input

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    """

    print('\nStack Intensity Regularisation')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        regularised_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='sir-img'))

        if overwrite is False \
            and os.path.isfile(regularised_file) :
                print("skip computation (use existing results)")
                output = {'result': regularised_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    sir = nighresjava.StackIntensityRegularisation()

    # set parameters
    
    # load image and use it to set dimensions and resolution
    img = load_volume(image)
    data = img.get_data()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    sir.setDimensions(dimensions[0], dimensions[1], dimensions[2])
       
    sir.setInputImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    if mask is not None:
        sir.setForegroundImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    # set algorithm parameters
    sir.setVariationRatio(float(cutoff))
    
    # execute the algorithm
    try:
        sir.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    regularised_data = np.reshape(np.array(sir.getRegularisedImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(regularised_data)
    header['cal_max'] = np.nanmax(regularised_data)
    regularised = nb.Nifti1Image(regularised_data, affine, header)

    if save_data:
        save_volume(regularised_file, regularised)
        return {'result': regularised_file}
    else:
        return {'result': regularised}
