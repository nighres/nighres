import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def shape_tensor_matching(ani, theta, images, references, img_types=None,
                            patch=3, search_dist=10.0, search_angle=5.0, threshold=0.2,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):
    """ Shape tensor mapping

    Estimates depth displacement from shape tensor images

    Parameters
    ----------
    ani: niimg
        Input 2D anisotropy image
    theta: niimg
        Input 2D angle image
    images: [niimg]
        Input 2D images for matching
    references: [niimg]
        Reference 2D images for matching
    img_types: [boolean]
        Image types for matching: True if angle, False if regular intensities 
        (default is all False)
    patch: int, optional 
        Maximum distance to define patch size (default is 3)
    search_dist: float, optional 
        Maximum distance to search (default is 10.0)
    search_angle: float, optional 
        Maximum distance to search (default is 5.0 degrees)
    threshold: float, optional
        Minimum anisotropy threshold for computation (default is 0.2)
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

        * result (niimg): The estimated distance offset

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    
    """

    print('\nShape Tensor Matching')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, ani)

        result_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=ani,
                                   suffix='stm-dz'))

        if overwrite is False \
            and os.path.isfile(result_file) :
                print("skip computation (use existing results)")
                output = {'result': result_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    stm = nighresjava.ShapeTensorMatching()

    # set parameters
    
    # load image and use it to set dimensions and resolution
    ani = load_volume(ani)
    data = ani.get_fdata()
    affine = ani.affine
    header = ani.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    stm.setDimensions(dimensions[0], dimensions[1], 1)
       
    stm.setInputAni(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    data = load_volume(theta).get_fdata()
    stm.setInputTheta(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    
    
    stm.setImageNumber(len(images))
    
    for idx,img in enumerate(images):
        data = load_volume(img).get_fdata()
        stm.setInputImageAt(idx,nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

        data = load_volume(references[idx]).get_fdata()
        stm.setReferenceImageAt(idx,nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
        
        if img_types is None:
            stm.setImageTypeAt(idx, False)
        else:
            stm.setImageTypeAt(idx, img_types[idx])

    # set algorithm parameters
    stm.setPatchSize(patch)
    
    stm.setSearchDistance(search_dist)
    stm.setSearchAngle(search_angle)
    
    stm.setAniThreshold(threshold)
    
    # execute the algorithm
    try:
        stm.execute2D()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    data = np.reshape(np.array(stm.getMatchingImage(),
                                    dtype=np.float32), newshape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(data)
    header['cal_max'] = np.nanmax(data)
    result = nb.Nifti1Image(data, affine, header)

    if save_data:
        save_volume(result_file, result)
        return {'result': result_file}
    else:
        return {'result': result}
