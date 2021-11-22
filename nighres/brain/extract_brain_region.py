import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
    _check_topology_lut_dir, _check_mgdm_atlas_file, _check_available_memory


def extract_brain_region(segmentation, levelset_boundary,
                         maximum_membership, maximum_label,
                         extracted_region, atlas_file=None,
                         normalize_probabilities=False,
                         estimate_tissue_densities=False,
                         partial_volume_distance=1.0,
                         save_data=False, overwrite=False, output_dir=None,
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
        Path to plain text atlas file (default is stored in DEFAULT_MGDM_ATLAS).
        or atlas name to be searched in MGDM_ATLAS_DIR
    extracted_region: {'left_cerebrum', 'right_cerebrum', 'cerebrum',
        'cerebellum', 'cerebellum_brainstem', 'subcortex', 'tissues(anat)',
        'tissues(func)', 'brain_mask'}
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
        (suffix of output files in brackets, # stands for shorthand names of
        the different extracted regions, respectively:
        rcr, lcr, cr, cb, cbs, sub, an, fn)

        * region_mask (niimg): Hard segmentation mask of the (GM) region
          of interest (_xmask-#gm)
        * inside_mask (niimg): Hard segmentation mask of the (WM) inside of
          the region of interest (_xmask-#wm)
        * background_mask (niimg): Hard segmentation mask of the (CSF) region
          background (_xmask-#bg)
        * region_proba (niimg): Probability map of the (GM) region
          of interest (_xproba-#gm)
        * inside_proba (niimg): Probability map of the (WM) inside of
          the region of interest (_xproba-#wm)
        * background_proba (niimg): Probability map of the (CSF) region
          background (_xproba-#bg)
        * region_lvl (niimg): Levelset surface of the (GM) region
          of interest (_xlvl-#gm)
        * inside_lvl (niimg): Levelset surface of the (WM) inside of
          the region of interest (_xlvl-#wm)
        * background_lvl (niimg): Levelset surface of the (CSF) region
          background (_xlvl-#bg)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    print('\nExtract Brain Region')

    # check atlas_file and set default if not given
    atlas_file = _check_mgdm_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, segmentation)

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    xbr = nighresjava.BrainExtractBrainRegion()

    # set parameters
    xbr.setAtlasFile(atlas_file)
    xbr.setExtractedRegion(extracted_region)
    xbr.setNormalizeProbabilities(normalize_probabilities)
    xbr.setEstimateTissueDensities(estimate_tissue_densities)
    xbr.setPartialVolumingDistance(partial_volume_distance)

	# build names for saving after setting the parameters to get the proper names
    if save_data:
        reg_mask_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=segmentation,
                                       suffix='xmask-'+xbr.getStructureName(), ))

        ins_mask_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=segmentation,
                                       suffix='xmask-'+xbr.getInsideName(), ))

        bg_mask_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                      rootfile=segmentation,
                                      suffix='xmask-'+xbr.getBackgroundName(), ))

        reg_proba_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                        rootfile=segmentation,
                                        suffix='xproba-'+xbr.getStructureName(), ))

        ins_proba_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                        rootfile=segmentation,
                                        suffix='xproba-'+xbr.getInsideName(), ))

        bg_proba_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=segmentation,
                                       suffix='xproba-'+xbr.getBackgroundName(), ))

        reg_lvl_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                      rootfile=segmentation,
                                      suffix='xlvl-'+xbr.getStructureName(), ))

        ins_lvl_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                      rootfile=segmentation,
                                      suffix='xlvl-'+xbr.getInsideName(), ))

        bg_lvl_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                     rootfile=segmentation,
                                     suffix='xlvl-'+xbr.getBackgroundName(), ))
        if overwrite is False \
            and os.path.isfile(reg_mask_file) \
            and os.path.isfile(ins_mask_file) \
            and os.path.isfile(bg_mask_file) \
            and os.path.isfile(reg_proba_file) \
            and os.path.isfile(ins_proba_file) \
            and os.path.isfile(bg_proba_file) \
            and os.path.isfile(reg_lvl_file) \
            and os.path.isfile(ins_lvl_file) \
            and os.path.isfile(bg_lvl_file) :

            print("skip computation (use existing results)")
            output = {'inside_mask': ins_mask_file,
                  'inside_proba': ins_proba_file,
                  'inside_lvl': ins_lvl_file,
                  'region_mask': reg_mask_file,
                  'region_proba': reg_proba_file,
                  'region_lvl': reg_lvl_file,
                  'background_mask': bg_mask_file,
                  'background_proba': bg_proba_file,
                  'background_lvl': bg_lvl_file}
            return output

    # load images and set dimensions and resolution
    seg = load_volume(segmentation)
    data = seg.get_data()
    affine = seg.affine
    header = seg.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    xbr.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    xbr.setResolutions(resolution[0], resolution[1], resolution[2])
    if (len(load_volume(maximum_membership).header.get_data_shape())>3):
        xbr.setComponents(load_volume(maximum_membership).header.get_data_shape()[3])
    else:
        xbr.setComponents(1)

    xbr.setSegmentationImage(nighresjava.JArray('int')(
        (data.flatten('F')).astype(int).tolist()))

    data = load_volume(levelset_boundary).get_data()
    xbr.setLevelsetBoundaryImage(nighresjava.JArray('float')(
        (data.flatten('F')).astype(float)))

    data = load_volume(maximum_membership).get_data()
    xbr.setMaximumMembershipImage(nighresjava.JArray('float')(
        (data.flatten('F')).astype(float)))

    data = load_volume(maximum_label).get_data()
    xbr.setMaximumLabelImage(nighresjava.JArray('int')(
        (data.flatten('F')).astype(int).tolist()))

    # execute
    try:
        xbr.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

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
        save_volume(ins_mask_file, inside_mask)
        save_volume(ins_proba_file, inside_proba)
        save_volume(ins_lvl_file, inside_lvl)
        save_volume(reg_mask_file, region_mask)
        save_volume(reg_proba_file, region_proba)
        save_volume(reg_lvl_file, region_lvl)
        save_volume(bg_mask_file, background_mask)
        save_volume(bg_proba_file, background_proba)
        save_volume(bg_lvl_file, background_lvl)

        output = {
            'inside_mask': ins_mask_file,
            'inside_proba': ins_proba_file,
            'inside_lvl': ins_lvl_file,
            'region_mask': reg_mask_file,
            'region_proba': reg_proba_file,
            'region_lvl': reg_lvl_file,
            'background_mask': bg_mask_file,
            'background_proba': bg_proba_file,
            'background_lvl': bg_lvl_file
        }
    else:
        output = {
            'inside_mask': inside_mask,
            'inside_proba': inside_proba,
            'inside_lvl': inside_lvl,
            'region_mask': region_mask,
            'region_proba': region_proba,
            'inside_lvl': region_lvl,
            'background_mask': background_mask,
            'background_proba': background_proba,
            'background_lvl': background_lvl
        }

    return output
