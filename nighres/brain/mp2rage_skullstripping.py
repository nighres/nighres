import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir


def mp2rage_skullstripping(second_inversion, t1_weighted=None, t1_map=None,
                           skip_zero_values=True, topology_lut_dir=None,
                           save_data=False, output_dir=None,
                           file_name=None):
    """ MP2RAGE skull stripping

    Estimates a brain mask from MRI data acquired with the MP2RAGE sequence.
    At least a T1-weighted or a T1 map image is required

    Parameters
    ----------
    second_inversion: niimg
        Second inversion image derived from MP2RAGE sequence
    t1_weighted: niimg
        T1-weighted image derived from MP2RAGE sequence (also referred to as
        "uniform" image)
        At least one of t1_weighted and t1_map is required
    t1_map: niimg
        Quantitative T1 map image derived from MP2RAGE sequence
        At least one of t1_weighted and t1_map is required
    skip_zero_values: bool
         Ignores voxels with zero value (default is True)
    topology_lut_dir: str, optional
        Path to directory in which topology files are stored (default is stored
        in TOPOLOGY_LUT_DIR)
    save_data: bool
        Save output data to file (default is False)
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

        * brain_mask (niimg): Binary brain mask (_strip_mask)
        * inv2_masked (niimg): Masked second inversion imamge (_strip_inv2)
        * t1w_masked (niimg): Masked T1-weighted image (_strip_t1w)
        * t1map_masked (niimg): Masked T1 map (_strip_t1map)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Details on the MP2RAGE
    sequence can be found in [1]_

    References
    ----------
    .. [1] Marques et al. (2010). MP2RAGE, a self bias-field corrected sequence
       for improved segmentation and T1-mapping at high field.
       DOI: 10.1016/j.neuroimage.2009.10.002
    """

    print('\nMP2RAGE Skull Stripping')

    # check topology lut dir and set default if not given
    topology_lut_dir = _check_topology_lut_dir(topology_lut_dir)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, second_inversion)

        inv2_file = _fname_4saving(file_name=file_name,
                                   rootfile=second_inversion,
                                   suffix='strip_inv2')
        mask_file = _fname_4saving(file_name=file_name,
                                   rootfile=second_inversion,
                                   suffix='strip_mask')
        if t1_weighted is not None:
            t1w_file = _fname_4saving(file_name=file_name,
                                      rootfile=t1_weighted,
                                      suffix='strip_t1w')

        if t1_map is not None:
            t1map_file = _fname_4saving(file_name=file_name,
                                        rootfile=t1_map,
                                        suffix='strip_t1map')

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass

    # create skulltripping instance
    stripper = cbstools.BrainMp2rageSkullStripping()

    # get dimensions and resolution from second inversion image
    inv2_img = load_volume(second_inversion)
    inv2_data = inv2_img.get_data()
    inv2_affine = inv2_img.get_affine()
    inv2_hdr = inv2_img.get_header()
    resolution = [x.item() for x in inv2_hdr.get_zooms()]
    dimensions = inv2_data.shape
    stripper.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    stripper.setResolutions(resolution[0], resolution[1], resolution[2])
    stripper.setSecondInversionImage(cbstools.JArray('float')(
                                    (inv2_data.flatten('F')).astype(float)))

    # pass other inputs
    if (t1_weighted is None and t1_map is None):
        raise ValueError('You must specify at least one of '
                         't1_weighted and t1_map')
    if t1_weighted is not None:
        t1w_img = load_volume(t1_weighted)
        t1w_data = t1w_img.get_data()
        t1w_affine = t1w_img.get_affine()
        t1w_hdr = t1w_img.get_header()
        stripper.setT1weightedImage(cbstools.JArray('float')(
                                      (t1w_data.flatten('F')).astype(float)))
    if t1_map is not None:
        t1map_img = load_volume(t1_map)
        t1map_data = t1map_img.get_data()
        t1map_affine = t1map_img.get_affine()
        t1map_hdr = t1map_img.get_header()
        stripper.setT1MapImage(cbstools.JArray('float')(
                                    (t1map_data.flatten('F')).astype(float)))

    stripper.setSkipZeroValues(skip_zero_values)
    stripper.setTopologyLUTdirectory(topology_lut_dir)

    # execute skull stripping
    try:
        stripper.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

    # collect outputs and potentially save
    inv2_masked_data = np.reshape(np.array(
                                stripper.getMaskedSecondInversionImage(),
                                dtype=np.float32), dimensions, 'F')
    inv2_hdr['cal_max'] = np.nanmax(inv2_masked_data)
    inv2_masked = nb.Nifti1Image(inv2_masked_data, inv2_affine, inv2_hdr)

    mask_data = np.reshape(np.array(stripper.getBrainMaskImage(),
                                    dtype=np.uint32), dimensions, 'F')
    inv2_hdr['cal_max'] = np.nanmax(mask_data)
    mask = nb.Nifti1Image(mask_data, inv2_affine, inv2_hdr)

    outputs = {'brain_mask': mask, 'inv2_masked': inv2_masked}

    if save_data:
        save_volume(os.path.join(output_dir, inv2_file), inv2_masked)
        save_volume(os.path.join(output_dir, mask_file), mask)

    if t1_weighted is not None:
        t1w_masked_data = np.reshape(np.array(
                                stripper.getMaskedT1weightedImage(),
                                dtype=np.float32), dimensions, 'F')
        t1w_hdr['cal_max'] = np.nanmax(t1w_masked_data)
        t1w_masked = nb.Nifti1Image(t1w_masked_data, t1w_affine, t1w_hdr)
        outputs['t1w_masked'] = t1w_masked

        if save_data:
            save_volume(os.path.join(output_dir, t1w_file), t1w_masked)

    if t1_map is not None:
        t1map_masked_data = np.reshape(np.array(
                                        stripper.getMaskedT1MapImage(),
                                        dtype=np.float32), dimensions, 'F')
        t1map_hdr['cal_max'] = np.nanmax(t1map_masked_data)
        t1map_masked = nb.Nifti1Image(t1map_masked_data, t1map_affine,
                                      t1map_hdr)
        outputs['t1map_masked'] = t1map_masked

        if save_data:
            save_volume(os.path.join(output_dir, t1map_file), t1map_masked)

    return outputs
