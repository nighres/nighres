import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def intensity_propagation(image, mask=None, combine='mean', distance_mm=5.0,
                      target='zero', scaling=1.0,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Intensity Propogation

    Propagates the values inside the mask (or non-zero) into the neighboring voxels

    Parameters
    ----------
    image: niimg
        Input image
    mask: niimg, optional
        Data mask to specify acceptable seeding regions
    combine: {'min','mean','max'}, optional
        Propagate using the mean (default), max or min data from neighboring voxels
    distance_mm: float, optional 
        Distance for the propagation (note: this algorithm will be slow for 
        large distances)
    target: {'zero','mask','lower','higher'}, optional
        Propagate into zero (default), masked out, lower or higher neighboring voxels
    scaling: float, optional
        Multiply the propagated values by a factor <=1 (default is 1)
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

        * result (niimg): The propagated intensity image

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    """

    print('\nIntensity Propagation')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        out_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='ppag-img'))

        if overwrite is False \
            and os.path.isfile(out_file) :
                print("skip computation (use existing results)")
                output = {'result': out_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    propag = nighresjava.IntensityPropagate()

    # set parameters
    
    # load image and use it to set dimensions and resolution
    img = load_volume(image)
    data = img.get_data()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    if len(dimensions)>2:
        propag.setDimensions(dimensions[0], dimensions[1], dimensions[2])
        propag.setResolutions(resolution[0], resolution[1], resolution[2])
    else:
        propag.setDimensions(dimensions[0], dimensions[1])
        propag.setResolutions(resolution[0], resolution[1])
        
    propag.setInputImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    
    if mask is not None:
        propag.setMaskImage(nighresjava.JArray('int')(
                (load_volume(mask).get_data().flatten('F')).astype(int).tolist()))
    
    # set algorithm parameters
    propag.setCombinationMethod(combine)
    propag.setPropagationDistance(distance_mm)
    propag.setTargetVoxels(target)
    propag.setPropogationScalingFactor(scaling)
    
    # execute the algorithm
    try:
        propag.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    propag_data = np.reshape(np.array(propag.getResultImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(propag_data)
    header['cal_max'] = np.nanmax(propag_data)
    out = nb.Nifti1Image(propag_data, affine, header)

    if save_data:
        save_volume(out_file, out)
        return {'result': out_file}
    else:
        return {'result': out}
