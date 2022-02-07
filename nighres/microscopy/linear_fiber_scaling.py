import numpy as np
import nibabel as nb
import os
import sys
import math
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, \
                    _check_available_memory


def linear_fiber_scaling(proba_image, theta_image, length_image, 
                              scaling=7, kept=5,
                              threshold=1e-9,
                              save_data=False, overwrite=False, output_dir=None,
                              file_name=None):

    """ Linear Fiber Scaling 

    Rescale extracted linear structures keeping multiple directions

    Parameters
    ----------
    proba_image: niimg
        Input probability image used to pick kept directions
    theta_image: niimg
        Input line angle image
    length_image: niimg
        Input line legnth image
    scaling: int
        Scaling factor for the rescaled images (default is 7)
    kept: int
        Number of kept directions
    threshold: float
        Detection threshold for grouping values (default is 1e-9)
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

        * proba (niimg): propagated probabilistic response (_lfs-proba)
        * theta (niimg): estimated line orientation angle (_lfs-theta)
        * length (niimg): estimated line length (_lfs-length)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    References
    ----------

    """

    print('\n Linear Fiber Scaling')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, proba_image)

        proba_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=proba_image,
                                  suffix='lfs-proba'))

        length_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=length_image,
                                  suffix='lfs-length'))

        theta_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=theta_image,
                                  suffix='lfs-theta'))

        if overwrite is False \
            and os.path.isfile(proba_file) \
            and os.path.isfile(length_file) \
            and os.path.isfile(theta_file):

            print("skip computation (use existing results)")
            output = {'proba': proba_file,
                      'length': length_file,
                      'theta': theta_file}
            return output


    # load input image and use it to set dimensions and resolution
    img = load_volume(proba_image)
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
    lfs = nighresjava.LinearFiberScaling()

    # set parameters
    lfs.setScaling(scaling)
    lfs.setNumberKept(kept)
    lfs.setDetectionThreshold(threshold)
    
    lfs.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    lfs.setResolutions(resolution[0], resolution[1], resolution[2])

    # input images
    lfs.setProbaImage(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))

    data = load_volume(theta_image).get_fdata()
    lfs.setThetaImage(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))

    data = load_volume(length_image).get_fdata()
    lfs.setLengthImage(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))

    # execute Extraction
    try:
        lfs.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # rescale dimensions
    if (dimensions[2]>1):
        print('4d result')
        dimensions = (math.ceil(dimensions[0]/scaling),math.ceil(dimensions[1]/scaling),dimensions[2],kept)
    else:
        print('3d result')
        dimensions = (math.ceil(dimensions[0]/scaling),math.ceil(dimensions[1]/scaling),kept)

    # reshape output to what nibabel likes
    proba_data = np.reshape(np.array(lfs.getScaledProbabilityImage(),
                                    dtype=np.float32), dimensions, 'F')

    length_data = np.reshape(np.array(lfs.getScaledLengthImage(),
                                   dtype=np.float32), dimensions, 'F')

    theta_data = np.reshape(np.array(lfs.getScaledAngleImage(),
                                    dtype=np.float32), dimensions, 'F')

 
    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(proba_data)
    proba_img = nb.Nifti1Image(proba_data, affine, header)

    header['cal_max'] = np.nanmax(length_data)
    length_img = nb.Nifti1Image(length_data, affine, header)

    header['cal_max'] = np.nanmax(theta_data)
    theta_img = nb.Nifti1Image(theta_data, affine, header)

    if save_data:
        save_volume(proba_file, proba_img)
        save_volume(length_file, length_img)
        save_volume(theta_file, theta_img)
       
        return {'proba': proba_file, 'length': length_file, 'theta': theta_file}
    else:
        return {'proba': proba_img, 'length': length_img, 'theta': theta_img}
