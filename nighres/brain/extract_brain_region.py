import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
    _check_topology_lut_dir, _check_atlas_file


def extract_brain_region(segmentation, levelset_boundary,
                         maximum_membership, maximum_label,
                         extracted_region, atlas_file=None,
                         normalize_probabilities=False,
                         estimate_tissue_densities=False,
                         partial_volume_distance=1.0,
                         save_data=False, output_dir=None,
                         file_name=None):
    """ Extract Brain Region

    Extracts masks, probability maps and levelset surfaces for specific brain
    regions and regions from a Multiple Object Geometric Deformable Model
    (MGDM) segmentation result.

    Parameters
    ----------
    segmentation: niimg
        Segmentation result from MGDM.
    levelset_boundary: niimg
        Levelset boundary from MGDM.
    maximum_membership: niimg
        4D image of the maximum membership values from MGDM.
    maximum_label: niimg
        4D imageof the maximum labels from MGDM.
    atlas_file: str, optional
        Path to plain text atlas file (default is stored in DEFAULT_ATLAS).
        or atlas name to be searched in ATLAS_DIR
    extracted_region: {'left_cerebrum', 'right_cerebrum', 'cerebrum', 'cerebellum', 'cerebellum_brainstem', 'subcortex', 'tissues(anat)', 'tissues(func)', 'brain_mask'}
        Region to be extracted from the MGDM segmentation.
    normalize_probabilities: bool
        Whether to normalize the output probabilities to sum to 1
        (default is False).
    estimate_tissue_densities: bool
        Wheter to recompute partial volume densities from the probabilites
        (slow, default is False).
    partial_volume_distance: float
        Distance in mm to use for tissues densities, if recomputed
        (default is 1mm).
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
        (suffix of output files in brackets, # stands for shorthand names of 
        the different extracted regions, respectively:
        rcr, lcr, cr, cb, cbs, sub, an, fn)

        * region_mask (niimg): Hard segmentation mask of the (GM) region
          of interest (_xmask_#gm)
        * inside_mask (niimg): Hard segmentation mask of the (WM) inside of
          the region of interest (_xmask_#wm)
        * background_mask (niimg): Hard segmentation mask of the (CSF) region
          background (_xmask_#bg)
        * region_proba (niimg): Probability map of the (GM) region
          of interest (_xproba_#gm)
        * inside_proba (niimg): Probability map of the (WM) inside of
          the region of interest (_xproba_#wm)
        * background_proba (niimg): Probability map of the (CSF) region
          background (_xproba_#bg)
        * region_lvl (niimg): Levelset surface of the (GM) region
          of interest (_xlvl_#gm)
        * inside_lvl (niimg): Levelset surface of the (WM) inside of
          the region of interest (_xlvl_#wm)
        * background_lvl (niimg): Levelset surface of the (CSF) region
          background (_xlvl_#bg)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    print('\nExtract Brain Region')

    # check atlas_file and set default if not given
    atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, segmentation)

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='8000m', maxheap='8000m')
    except ValueError:
        pass
    # create algorithm instance
    xbr = cbstools.BrainExtractBrainRegion()

    # set parameters
    xbr.setAtlasFile(atlas_file)
    xbr.setExtractedRegion(extracted_region)
    xbr.setNormalizeProbabilities(normalize_probabilities)
    xbr.setEstimateTissueDensities(estimate_tissue_densities)
    xbr.setPartialVolumingDistance(partial_volume_distance)

    # load images and set dimensions and resolution
    seg = load_volume(segmentation)
    data = seg.get_data()
    affine = seg.get_affine()
    header = seg.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    xbr.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    xbr.setResolutions(resolution[0], resolution[1], resolution[2])
    xbr.setComponents(load_volume(maximum_membership).get_header().get_data_shape()[3])

    xbr.setSegmentationImage(cbstools.JArray('int')(
        (data.flatten('F')).astype(int)))

    data = load_volume(levelset_boundary).get_data()
    xbr.setLevelsetBoundaryImage(cbstools.JArray('float')(
        (data.flatten('F')).astype(float)))

    data = load_volume(maximum_membership).get_data()
    xbr.setMaximumMembershipImage(cbstools.JArray('float')(
        (data.flatten('F')).astype(float)))

    data = load_volume(maximum_label).get_data()
    xbr.setMaximumLabelImage(cbstools.JArray('int')(
        (data.flatten('F')).astype(int)))

    # execute
    try:
        xbr.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

	# build names for saving after the computations to get the proper names
    if save_data:
        reg_mask_file = _fname_4saving(file_name=file_name,
                                       rootfile=segmentation,
                                       suffix='xmask'+xbr.getStructureName(), )

        ins_mask_file = _fname_4saving(file_name=file_name,
                                       rootfile=segmentation,
                                       suffix='xmask'+xbr.getInsideName(), )

        bg_mask_file = _fname_4saving(file_name=file_name,
                                      rootfile=segmentation,
                                      suffix='xmask'+xbr.getBackgroundName(), )

        reg_proba_file = _fname_4saving(file_name=file_name,
                                        rootfile=segmentation,
                                        suffix='xproba'+xbr.getStructureName(), )

        ins_proba_file = _fname_4saving(file_name=file_name,
                                        rootfile=segmentation,
                                        suffix='xproba'+xbr.getInsideName(), )

        bg_proba_file = _fname_4saving(file_name=file_name,
                                       rootfile=segmentation,
                                       suffix='xproba'+xbr.getBackgroundName(), )

        reg_lvl_file = _fname_4saving(file_name=file_name,
                                      rootfile=segmentation,
                                      suffix='xlvl'+xbr.getStructureName(), )

        ins_lvl_file = _fname_4saving(file_name=file_name,
                                      rootfile=segmentation,
                                      suffix='xlvl'+xbr.getInsideName(), )

        bg_lvl_file = _fname_4saving(file_name=file_name,
                                     rootfile=segmentation,
                                     suffix='xlvl'+xbr.getBackgroundName(), )


    # inside region
    # reshape output to what nibabel likes
    mask_data = np.reshape(np.array(xbr.getInsideWMmask(),
                                    dtype=np.int32), dimensions, 'F')

    proba_data = np.reshape(np.array(xbr.getInsideWMprobability(),
                                     dtype=np.float32), dimensions, 'F')

    lvl_data = np.reshape(np.array(xbr.getInsideWMlevelset(),
                                   dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(mask_data)
    header['cal_max'] = np.nanmax(mask_data)
    inside_mask = nb.Nifti1Image(mask_data, affine, header)

    header['cal_min'] = np.nanmin(proba_data)
    header['cal_max'] = np.nanmax(proba_data)
    inside_proba = nb.Nifti1Image(proba_data, affine, header)

    header['cal_min'] = np.nanmin(lvl_data)
    header['cal_max'] = np.nanmax(lvl_data)
    inside_lvl = nb.Nifti1Image(lvl_data, affine, header)

    # main region
    # reshape output to what nibabel likes
    mask_data = np.reshape(np.array(xbr.getStructureGMmask(),
                                    dtype=np.int32), dimensions, 'F')

    proba_data = np.reshape(np.array(xbr.getStructureGMprobability(),
                                     dtype=np.float32), dimensions, 'F')

    lvl_data = np.reshape(np.array(xbr.getStructureGMlevelset(),
                                   dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(mask_data)
    header['cal_max'] = np.nanmax(mask_data)
    region_mask = nb.Nifti1Image(mask_data, affine, header)

    header['cal_min'] = np.nanmin(proba_data)
    header['cal_max'] = np.nanmax(proba_data)
    region_proba = nb.Nifti1Image(proba_data, affine, header)

    header['cal_min'] = np.nanmin(lvl_data)
    header['cal_max'] = np.nanmax(lvl_data)
    region_lvl = nb.Nifti1Image(lvl_data, affine, header)

    # background region
    # reshape output to what nibabel likes
    mask_data = np.reshape(np.array(xbr.getBackgroundCSFmask(),
                                    dtype=np.int32), dimensions, 'F')

    proba_data = np.reshape(np.array(xbr.getBackgroundCSFprobability(),
                                     dtype=np.float32), dimensions, 'F')

    lvl_data = np.reshape(np.array(xbr.getBackgroundCSFlevelset(),
                                   dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(mask_data)
    header['cal_max'] = np.nanmax(mask_data)
    background_mask = nb.Nifti1Image(mask_data, affine, header)

    header['cal_min'] = np.nanmin(proba_data)
    header['cal_max'] = np.nanmax(proba_data)
    background_proba = nb.Nifti1Image(proba_data, affine, header)

    header['cal_min'] = np.nanmin(lvl_data)
    header['cal_max'] = np.nanmax(lvl_data)
    background_lvl = nb.Nifti1Image(lvl_data, affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, ins_mask_file), inside_mask)
        save_volume(os.path.join(output_dir, ins_proba_file), inside_proba)
        save_volume(os.path.join(output_dir, ins_lvl_file), inside_lvl)
        save_volume(os.path.join(output_dir, reg_mask_file), region_mask)
        save_volume(os.path.join(output_dir, reg_proba_file), region_proba)
        save_volume(os.path.join(output_dir, reg_lvl_file), region_lvl)
        save_volume(os.path.join(output_dir, bg_mask_file), background_mask)
        save_volume(os.path.join(output_dir, bg_proba_file), background_proba)
        save_volume(os.path.join(output_dir, bg_lvl_file), background_lvl)

    return {'inside_mask': inside_mask, 'inside_proba': inside_proba,
            'inside_lvl': inside_lvl, 'region_mask': region_mask,
            'region_proba': region_proba, 'inside_lvl': region_lvl,
            'background_mask': background_mask,
            'background_proba': background_proba,
            'background_lvl': background_lvl}
