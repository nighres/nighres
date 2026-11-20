import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, \
                    _check_available_memory


def oct_multiview_combination(input_images, orient_images,
                              angles, 
                              save_data=False, overwrite=False, output_dir=None,
                              file_name=None):

    """ OCT multi-view combination

    Extracts OCT directionality from birefringence and oreintation in multiple
    co-registered views at different angles (assuming the rotation along the Y axis)


    Parameters
    ----------
    input_images: [niimg]
        Input birefringence images to combine
    orient_images: [niimg]
        Input orientation images to combine
    angles:[float]
        Angles between the different views and the origin in degrees
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

        * result (niimg): computed direction image (_omc-dir)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Based on the methods described 
    in [1]_ and additional geometric modeling.

    References
    ----------
    .. [1] Liu et al (2023), Quantitative imaging of three-dimensional fiber
    orientation in the human brain via two illumination angles using 
    polarization-sensitive optical coherence tomography, bioRxiv.

    """

    print('\n OCT contrast integration')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_images[0])

        result_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_images[0],
                                  suffix='omc-dir'))

        if overwrite is False \
            and os.path.isfile(result_file) :

            print("skip computation (use existing results)")
            return {'result': result_file}


    # load input image and use it to set dimensions and resolution
    img = load_volume(input_images[0])
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
    omc = nighresjava.OctMultiviewCombination()

    # set parameters
    if len(input_images)!=len(orient_images) or len(orient_images)!=len(angles) or len(angles)!=len(input_images):
        print("images, orientations, and angles do not match, aborting")
        return
    
    omc.setImageNumber(len(input_images))
    
    omc.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    omc.setResolutions(resolution[0], resolution[1], resolution[2])

    # input images
    for idx,img in enumerate(input_images):
        omc.setInputImageAt(idx,nighresjava.JArray('float')(
                                            (load_volume(img).get_fdata().flatten('F')).astype(float)))

    for idx,img in enumerate(orient_images):
        omc.setOrientImageAt(idx,nighresjava.JArray('float')(
                                            (load_volume(img).get_fdata().flatten('F')).astype(float)))

    for idx,theta in enumerate(angles):
        omc.setAngleAt(idx,theta/180.0*np.pi)


    # execute Extraction
    try:
        omc.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return
    
    dim4d = (dimensions[0],dimensions[1],dimensions[2],3)
    
    # reshape output to what nibabel likes
    result_data = np.reshape(np.array(omc.getDirectionImage(),
                                    dtype=np.float32), shape=dim4d, order='F')


    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(result_data)
    result_img = nb.Nifti1Image(result_data, affine, header)

    if save_data:
        save_volume(result_file, result_img)
        return {'result': result_file}
    else:
        return {'result': result_img}
