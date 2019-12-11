import os
import sys
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, _check_available_memory


def probability_to_levelset(probability_image, mask_image=None,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):

    """Levelset from probability map

    Creates a levelset surface representations from a probabilistic map
    or a mask. The levelset indicates each voxel's distance to the closest
    boundary. It takes negative values inside and positive values outside
    of the object.

    Parameters
    ----------
    probability_image: niimg
        Probability image to be turned into levelset. Values should be in
        [0, 1], either a binary mask or defining the boundary at 0.5.
    mask_image: niimg, optional
        Mask image defining the region in which to compute the levelset. Values
        equal to zero are set to maximum distance.
    save_data: bool, optional
        Save output data to file (default is False)
    overwrite: bool, optional
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

        * result (niimg): Levelset representation of surface (_p2l-surf)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin
    """

    print("\nProbability to Levelset")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, probability_image)

        levelset_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=probability_image,
                                       suffix='p2l-surf'))

        if overwrite is False \
            and os.path.isfile(levelset_file) :

            print("skip computation (use existing results)")
            output = {'result': levelset_file}
            return output

    # start virtual machine if not running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initiate class
    prob2level = nighresjava.SurfaceProbabilityToLevelset()

    # load the data
    prob_img = load_volume(probability_image)
    prob_data = prob_img.get_data()
    hdr = prob_img.header
    aff = prob_img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = prob_data.shape

    # set parameters from input data
    prob2level.setProbabilityImage(nighresjava.JArray('float')(
                                    (prob_data.flatten('F')).astype(float)))
    
    if (mask_image is not None):
        mask_data = load_volume(mask_image).get_data()
        prob2level.setMaskImage(nighresjava.JArray('int')(
                                    (mask_data.flatten('F')).astype(int).tolist()))
        
    if len(dimensions)>2:
        prob2level.setResolutions(resolution[0], resolution[1], resolution[2])
        prob2level.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    else:
        prob2level.setResolutions(resolution[0], resolution[1], 1.0)
        prob2level.setDimensions(dimensions[0], dimensions[1], 1)

    # execute class
    try:
        prob2level.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # collect outputs
    levelset_data = np.reshape(np.array(prob2level.getLevelSetImage(),
                               dtype=np.float32), dimensions, 'F')

    hdr['cal_max'] = np.nanmax(levelset_data)
    levelset = nb.Nifti1Image(levelset_data, aff, hdr)

    if save_data:
        save_volume(levelset_file, levelset)
        return {'result': levelset_file}
    else:
        return {'result': levelset}
