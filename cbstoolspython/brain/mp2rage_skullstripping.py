import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving

# TODO: set default LUT dir
TOPOLOGY_LUT_DIR = '/home/julia/workspace/cbstools-python/lut/'


def mp2rage_skullstripping(t1_weighted=None, t1_map=None,
                           second_inversion=None, topology_lut_dir=None,
                           save_data=False, output_dir=None,
                           file_name=None, file_extension=None):
    """ MP2RAGE skull stripping

    Estimate a brain mask from MRI data acquired with the MP2RAGE sequence [1].
    At least a T1-weighted or a T1 map image is required

    Parameters
    ----------
    t1_weighted: TODO:type
        T1-weighted image derived from MP2RAGE sequence (also referred to as
        "uniform" image) to perform skullstripping on.
        At least one of t1_weighted and t1_map is required
    t1_map: TODO:type
        Quantitative T1 map image derived from MP2RAGE sequence to perform
        skullstripping on.
        At least one of t1_weighted and t1_map is required
    second_inversion: TODO:type, optional
        Second inversion image derived from MP2RAGE sequence to aid
        skullstripping.
    topology_lut_dir: str, optional
        Path to directory in which topology files are stored (default is stored
        in TOPOLOGY_LUT_DIR)
    save_data: bool
        Save output data to file (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_extension: str, optional
        Desired extension for output files (determines file type)

    Returns
    ----------
    tuple (dictionary?)
        brain_mask
        masked_t1w
        masked_t1map
        masked_inv2

    References
    ----------
    [1] Marques et al. (2010). MP2RAGE, a self bias-field corrected sequence
    for improved segmentation and T1-mapping at high field.
    DOI: 10.1016/j.neuroimage.2009.10.002
    """

    # make sure one of t1_weighted or t1_map is given and set one as
    # main image for dimensions, resolution and base for saving
    if (t1_weighted is None and t1_map is None):
        raise ValueError('You must specify at least one of '
                         't1_weighted and t1_map')

    # set default topology lut dir if not given
    if topology_lut_dir is None:
        topology_lut_dir = TOPOLOGY_LUT_DIR
    else:
        # if we don't end in a path sep, we need to make sure that we add it
        if not(topology_lut_dir[-1] == os.path.sep):
            topology_lut_dir += os.path.sep

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass

    # create skulltripping instance
    stripper = cbstools.BrainMp2rageSkullStripping()

    # pass input images to skull stripper class
    if t1_weighted is not None:
        img = load_volume(t1_weighted)
        data = img.get_data()
        stripper.setT1weightedImage(cbstools.JArray('float')(
                                            (data.flatten('F')).astype(float)))
        if t1_map is not None:
            data2 = load_volume(t1_map).get_data()
            stripper.setT1MapImage(cbstools.JArray('float')(
                                        (data2.flatten('F')).astype(float)))

    elif t1_weighted is None and t1_map is not None:
        img = load_volume(t1_map)
        data = img.get_data()
        stripper.setT1MapImage(cbstools.JArray('float')(
                                    (data.flatten('F')).astype(float)))

    # set dimensions and resolutions from input
    # (whichever given, stored in variables img and data above
    affine = img.get_affine()
    header = img.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    stripper.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    stripper.setResolutions(resolution[0], resolution[1], resolution[2])

    # skip zero values?
    # stripper.setSkipZeroValues

    # execute skull stripping
    try:
        print("Running skull stripping")
        stripper.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

    # collect outputs and potentially save
    # TODO: also do the header max recalculation?
    if save_data:
        if t1_weighted is not None:
            output_dir = _output_dir_4saving(output_dir, t1_weighted)
        else:
            output_dir = _output_dir_4saving(output_dir, t1_map)
        print("\n Saving outputs to {0}".format(output_dir))

    if t1_weighted is not None:
            t1w_masked = nb.Nifti1Image(
                            np.reshape(np.array(
                                    mgdm.stripper.getMaskedT1weightedImage(),
                                    dtype=np.float32), dimensions, 'F'),
                            affine, header)
            if save_data:
                t1w_file = _fname_4saving(rootfile=t1_weighted,
                                          suffix='masked',
                                          extension=file_extension)
                mask_file = _fname_4saving(rootfile=t1_weighted, suffix='mask',
                                           extension=file_extension)

                save_volume(os.path.join(output_dir, t1w_file), t1w_masked)

    if t1_map is not None:
            t1map_masked = nb.Nifti1Image(
                                np.reshape(np.array(
                                      mgdm.stripper.getMaskedT1mapImage(),
                                      dtype=np.float32), dimensions, 'F'),
                                affine, header)
            if save_data:
                t1map_file = _fname_4saving(rootfile=t1_map, suffix='masked',
                                            extension=file_extension)
                if t1_weighted is None:
                    mask_file = _fname_4saving(rootfile=t1_map, suffix='mask',
                                               extension=file_extension)

                save_volume(os.path.join(output_dir, t1map_file), t1map_masked)

    if second_inversion is not None:
            inv2_masked = nb.Nifti1Image(
                            np.reshape(np.array(
                                mgdm.stripper.getMaskedSecondINversionImage(),
                                dtype=np.float32), dimensions, 'F'),
                            affine, header)
            if save_data:
                inv2_file = _fname_4saving(rootfile=second_inversion,
                                           suffix='masked',
                                           extension=file_extension)
                save_volume(os.path.join(output_dir, inv2_file), inv2_masked)

    mask = nb.Nifti1Image(
                np.reshape(np.array(mgdm.stripper.getBrainMaskImage(),
                           dtype=np.uint32), dimensions, 'F'),
                affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, mask_file), mask)

    return
