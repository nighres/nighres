import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, \
                    _check_available_memory


def oct_contrast_integration(input_image, weight_image,
                              contrast='birefringence', 
                              weighting='linear',
                              threshold=0.1,
                              save_data=False, overwrite=False, output_dir=None,
                              file_name=None):

    """ OCT contrast integration

    Extracts OCT features from a Z stack image.


    Parameters
    ----------
    input_image: niimg
        Input image to process
    weight_image: niimg
        Input weight image to use, typically intensity
    contrast: {'birefringence','orientation'}
        Which contrast processing to do
    weighting: {'constant','linear','log'}
        Which form of weighting to use (default is 'linear')
    threshold:float
        Where to threshold the weights (default is 0.1)
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

        * result (niimg): computed contrast image (_oci-biref or _oci-orient)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Based on the methods described 
    in [1]_.

    References
    ----------
    .. [1] Liu et al (2023), Quantitative imaging of three-dimensional fiber
    orientation in the human brain via two illumination angles using 
    polarization-sensitive optical coherence tomography, bioRxiv.

    """

    print('\n OCT contrast integration')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_image)

        if contrast=="birefringence":
            result_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='oci-biref'))
        elif contrast=="orientation":
            result_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='oci-orient'))
        else:
            result_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='oci-img'))

        if overwrite is False \
            and os.path.isfile(result_file) :

            print("skip computation (use existing results)")
            return {'result': result_file}


    # load input image and use it to set dimensions and resolution
    img = load_volume(input_image)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    if (len(dimensions)<3): dimensions = (dimensions[0], dimensions[1], 1)
    if (len(resolution)<3): resolution = [resolution[0], resolution[1], 1.0]

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create extraction instance
    oci = nighresjava.OctContrastIntegration()

    # set parameters
    oci.setContrastType(contrast)
    oci.setWeightingType(weighting)
    oci.setRatio(threshold)
    
    oci.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    oci.setResolutions(resolution[0], resolution[1], resolution[2])

    # input input_image
    oci.setInputImage(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))

    oci.setWeightImage(nighresjava.JArray('float')(
                                            (load_volume(weight_image).get_fdata().flatten('F')).astype(float)))

    # execute Extraction
    try:
        oci.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return
    
    dim2d = (dimensions[0],dimensions[1])
    
    # reshape output to what nibabel likes
    result_data = np.reshape(np.array(oci.getContrastImage(),
                                    dtype=np.float32), shape=dim2d, order='F')


    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(result_data)
    result_img = nb.Nifti1Image(result_data, affine, header)

    if save_data:
        save_volume(result_file, result_img)
        return {'result': result_file}
    else:
        return {'result': result_img}
