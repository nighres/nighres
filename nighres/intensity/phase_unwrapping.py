import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def phase_unwrapping(image, mask=None, nquadrants=3, rescale_phs=True,
                      tv_flattening=False, tv_scale=0.5,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Fast marching phase unwrapping

    Fast marching method for unwrapping phase images, based on _[1]

    Parameters
    ----------
    image: niimg
        Input phase image to unwrap
    mask: niimg, optional
        Data mask to specify acceptable seeding regions
    nquadrants: int, optional
        Number of image quadrants to use (default is 3)
    rescale_phs: bool, optional
        Whether to rescale the phase data of keep it as is, assuming radians
        (default is True)
    tv_flattening: bool, optional 
        Whether or not to run a post-processing step to remove background
        phase variations with a total variation filter (default is False)
    tv_scale: float, optional
        Relative intensity scale for the TV filter (default is 0.5)
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

        * result (niimg): The unwrapped image rescaled in radians

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Algorithm adapted from [1]_
    with additional seeding in multiple image quadrants to reduce the effects
    of possible phase singularities

    References
    ----------
    .. [1] Abdul-Rahman, Gdeisat, Burton and Lalor. Fast three-dimensional 
           phase-unwrapping algorithm based on sorting by reliability following 
           a non-continuous path. doi: 10.1117/12.611415
    """

    print('\nFast marching phase unwrapping')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        out_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='unwrap-img'))

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
    unwrap = nighresjava.FastMarchingPhaseUnwrapping()

    # set parameters
    
    # load image and use it to set dimensions and resolution
    img = load_volume(image)
    data = img.get_data()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    dimensions3D = (dimensions[0], dimensions[1], dimensions[2])

    unwrap.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    unwrap.setResolutions(resolution[0], resolution[1], resolution[2])

    unwrap.setPhaseImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)[0:dimensions[0]*dimensions[1]*dimensions[2]]))
    
    
    if mask is not None:
        unwrap.setMaskImage(idx, nighresjava.JArray('int')(
                (load_volume(mask).get_data().flatten('F')).astype(int).tolist()))
    
    # set algorithm parameters
    unwrap.setQuadrantNumber(nquadrants)
    unwrap.setRescalePhase(rescale_phs)
    if tv_flattening: unwrap.setTVPostProcessing("TV-residuals")
    unwrap.setTVScale(tv_scale)
    
    # execute the algorithm
    try:
        unwrap.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    unwrap_data = np.reshape(np.array(unwrap.getCorrectedImage(),
                                    dtype=np.float32), dimensions3D, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(unwrap_data)
    header['cal_max'] = np.nanmax(unwrap_data)
    out = nb.Nifti1Image(unwrap_data, affine, header)

    if save_data:
        save_volume(out_file, out)
        return {'result': out_file}
    else:
        return {'result': out}
