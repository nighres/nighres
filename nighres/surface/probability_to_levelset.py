import os
import numpy as np
import nibabel as nb
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving


def probability_to_levelset(probability_image,
                            save_data=False, output_dir=None,
                            file_name=None):

    """Levelset from tissue classification

    Creates a levelset surface representations from a probabilistic or
    deterministic tissue classification. The levelset indicates each voxel's
    distance to the closest boundary. It takes negative values inside and
    positive values outside of the brain.

    Parameters
    ----------
    probability_image: niimg
        Tissue segmentation to be turned into levelset. Values should be in
        [0, 1], either a binary mask or defining the boundary at 0.5.
    save_data: bool
        Save output data to file (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)

    Returns
    ----------
    niimg
        Levelset representation of surface (output file suffix _levelset)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin
    """

    print("\nProbability to Levelset")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, probability_image)

        levelset_file = _fname_4saving(file_name=file_name,
                                       rootfile=probability_image,
                                       suffix='levelset')

    # start virtual machine if not running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass

    # initiate class
    prob2level = cbstools.SurfaceProbabilityToLevelset()

    # load the data
    prob_img = load_volume(probability_image)
    prob_data = prob_img.get_data()
    hdr = prob_img.get_header()
    aff = prob_img.get_affine()
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = prob_data.shape

    # set parameters from input data
    prob2level.setProbabilityImage(cbstools.JArray('float')(
                                    (prob_data.flatten('F')).astype(float)))
    prob2level.setResolutions(resolution[0], resolution[1], resolution[2])
    prob2level.setDimensions(dimensions[0], dimensions[1], dimensions[2])

    # execute class
    try:
        prob2level.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

    # collect outputs
    levelset_data = np.reshape(np.array(prob2level.getLevelSetImage(),
                               dtype=np.float32), dimensions, 'F')

    hdr['cal_max'] = np.nanmax(levelset_data)
    levelset = nb.Nifti1Image(levelset_data, aff, hdr)

    if save_data:
        save_volume(os.path.join(output_dir, levelset_file), levelset)

    return levelset
