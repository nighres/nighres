import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def linear_fiber_interpolation(image, references, mapped_proba, mapped_theta, 
                            mapped_lambda, mapped_dim=1, weights = None,
                            patch=2, search=3, median=True,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):
    """ Linear fiber interpolation

    Uses a simple non-local means approach adapted from [1]_
    to interpolate extracted line information across slices

    Parameters
    ----------
    image: niimg
        Input 2D image
    references: [niimg]
        Reference 2D images to use for intensity mapping
    mapped_proba: [niimg]
        Corresponding mapped 3D images to use for line probabilites
    mapped_theta: [niimg]
        Corresponding mapped 3D images to use for line directions
    mapped_lambda: [niimg]
        Corresponding mapped 3D images to use for line lengths
    mapped_dim: int
        Thrid dimension of the mapped 3D images (default is 1)
    weights: [float], optional
        Weight factors for the 2D images (default is 1 for all)
    patch: int, optional 
        Maximum distance to define patch size (default is 2)
    search: int, optional 
        Maximum distance to define search window size (default is 3)
    median: bool
        Whether to use median instead of mean of the patches (default is True)
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

        * proba (niimg): The probability mapped input
        * theta (niimg): The direction mapped input
        * lambda (niimg): The length mapped input

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.


    References
    ----------
    .. [1] P. Coupé, J.V. Manjón, V. Fonov, J. Pruessner, M. Robles, D.L. Collins,
       Patch-based segmentation using expert priors: Application to hippocampus 
       and ventricle msegmentation, NeuroImage, vol. 54, pp. 940--954, 2011.
    """

    print('\nLinear fiber interpolation')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='lfi-proba'))

        theta_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='lfi-theta'))

        lambda_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='lfi-lambda'))

        if overwrite is False \
            and os.path.isfile(proba_file) \
            and os.path.isfile(theta_file) \
            and os.path.isfile(lambda_file) :
                print("skip computation (use existing results)")
                output = {'proba': proba_file, 'theta': theta_file, 'lambda': lambda_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    lfi = nighresjava.NonlocalLinearFiberInterpolation()

    # set parameters
    
    # load image and use it to set dimensions and resolution
    img = load_volume(image)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    lfi.setDimensions(dimensions[0], dimensions[1], 1)
    lfi.setLineNumber(mapped_dim)
       
    lfi.setInputImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    lfi.setReferenceNumber(len(references))
    
    for idx,ref in enumerate(references):
        data = load_volume(ref).get_fdata()
        lfi.setReferenceImageAt(idx,nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

        data = load_volume(mapped_proba[idx]).get_fdata()
        lfi.setMappedProbaAt(idx,nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

        data = load_volume(mapped_theta[idx]).get_fdata()
        lfi.setMappedThetaAt(idx,nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

        data = load_volume(mapped_lambda[idx]).get_fdata()
        lfi.setMappedLambdaAt(idx,nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

        if weights is not None:
            lfi.setWeightAt(idx, weights[idx])
        else:
            lfi.setWeightAt(idx, 1.0)
            
    # set algorithm parameters
    lfi.setPatchDistance(patch)
    lfi.setSearchDistance(search)
    lfi.setUseMedian(median)
    
    # execute the algorithm
    try:
        lfi.execute2D()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    dimensions = (dimensions[0],dimensions[1],mapped_dim)
    data = np.reshape(np.array(lfi.getMappedProba(),
                                    dtype=np.float32), shape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(data)
    header['cal_max'] = np.nanmax(data)
    result_proba = nb.Nifti1Image(data, affine, header)

    # reshape output to what nibabel likes
    dimensions = (dimensions[0],dimensions[1],mapped_dim)
    data = np.reshape(np.array(lfi.getMappedTheta(),
                                    dtype=np.float32), shape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(data)
    header['cal_max'] = np.nanmax(data)
    result_theta = nb.Nifti1Image(data, affine, header)

    # reshape output to what nibabel likes
    dimensions = (dimensions[0],dimensions[1],mapped_dim)
    data = np.reshape(np.array(lfi.getMappedLambda(),
                                    dtype=np.float32), shape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(data)
    header['cal_max'] = np.nanmax(data)
    result_lambda = nb.Nifti1Image(data, affine, header)

    if save_data:
        save_volume(proba_file, result_proba)
        save_volume(theta_file, result_theta)
        save_volume(lambda_file, result_lambda)
        return {'proba': proba_file, 'theta': theta_file, 'lambda': lambda_file}
    else:
        return {'proba': result_proba, 'theta': result_theta, 'lambda': result_lambda}
