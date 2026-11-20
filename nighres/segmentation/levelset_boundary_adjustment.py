import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def levelset_boundary_adjustment(levelset, image, mask=None, distance=3.0, spread=1.0,
                      contrast="increasing", iterations=10, repeats=2, smoothness=0.5,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Levelset Boundary Adjustment

    Refines a segmentation result to match the image contrast gradient, by fitting a sigmoid model.

    Parameters
    ----------
    levelset: niimg
        Input levelset representation of the region of interest
    image: niimg
        Input image defining the contrast to use
    mask: niimg, optional
        Data mask to specify acceptable seeding regions
    distance: float, optional
        Distance to the boundary to include in the modeling (default is 3.0)
    spread: float, optional
        Distance to use along the boundary to define the local sigmoid fit (default is 1.0)
    contrast: string, optional
        Type of contrast to use: increasing, decreasing, ridge, etc (default is increasing)
    iterations: int, optional
        Number of iterations for the adjustment (default is 10)
    repeats: int, optional
        Number of repeats of the full adjustment and levelset deformation procedure (default is 2)
    smoothness: float, optional
        Amount of curvature smoothing to use when deforming the levelset, in [0,1] (default is 0.5)
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

        * result (niimg): The adjusted levelset image
        * proba (niimg): The adjustment confidence probability
    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.    

    """

    print('\nLevelset boundary adjustment')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, levelset)

        result_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=levelset,
                                   suffix='lba-lvl'))

        proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=levelset,
                                   suffix='lba-proba'))

        if overwrite is False \
            and os.path.isfile(result_file) \
            and os.path.isfile(proba_file) :
                print("skip computation (use existing results)")
                output = {'result': result_file, 'proba': proba_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    algo = nighresjava.LevelsetBoundaryAdjustment()

    # set parameters
    
    # load image and use it to set dimensions and resolution
    img = load_volume(image)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    algo.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    algo.setResolutions(resolution[0], resolution[1], resolution[2])
        
    algo.setContrastImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    algo.setLevelsetImage(nighresjava.JArray('float')(
                (load_volume(levelset).get_fdata().flatten('F')).astype(float)))
    
    if mask is not None:
        algo.setMaskImage(nighresjava.JArray('int')(
                (load_volume(mask).get_fdata().flatten('F')).astype(int).tolist()))
    
    # set algorithm parameters
    algo.setBoundaryDistance(distance)
    algo.setLocalSpread(spread)
    algo.setContrastType(contrast)
    algo.setIterations(iterations)
    algo.setRepeats(repeats)
    algo.setSmoothness(smoothness)
    
    # execute the algorithm
    try:
        algo.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    result_data = np.reshape(np.array(algo.getLevelsetImage(),
                                    dtype=np.float32), shape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(result_data)
    header['cal_max'] = np.nanmax(result_data)
    result = nb.Nifti1Image(result_data, affine, header)

    proba_data = np.reshape(np.array(algo.getProbaImage(),
                                    dtype=np.float32), shape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(proba_data)
    header['cal_max'] = np.nanmax(proba_data)
    proba = nb.Nifti1Image(proba_data, affine, header)

    if save_data:
        save_volume(result_file, result)
        save_volume(proba_file, proba)
        return {'result': result_file, 'proba': proba_file}
    else:
        return {'result': result, 'proba': proba}
