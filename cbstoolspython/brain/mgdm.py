# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:41:28 2016
@author: Christopher J. Steele
steele{AT}cbs{dot}mpg{dot}de
"""

import numpy as np
import nibabel as nb
import os
import cbstools
from ..io import load_volume, save_volume

# TODO
ATLAS_DIR = '/home/julia/workspace/cbstools-python/atlases/brain-segmentation-prior3.0/'
TOPOLOGY_LUT_DIR = '/home/julia/workspace/cbstools-python/lut/'
DEFAULT_ATLAS = "brain-atlas-3.0.3.txt"


###### utils? #####
def create_dir(some_directory):
    """
    Create directory recursively if it does not exist
      - uses os.mkdirs
    """
    import os
    if not os.path.exists(some_directory):
        os.makedirs(some_directory)


def normalise(img_d):
    return (img_d - np.min(img_d)) / np.max(img_d)

# orientation of the x, y, z
def get_affine_orientation(img):
    return nb.io_orientation(img)


def flip_affine_data_orientation(  # d,
                                 a, flip_LR=False,
                                 flip_AP=False, flip_IS=False):
    if flip_LR:
        a[1, 1] = a[1, 1] * -1
    if flip_AP:
        a[2, 2] = a[2, 2] * -1
        # d=d[:,::-1,:]
    if flip_IS:
        a[3, 3] = a[3, 3] * -1
    # return d, a
    return a


#### MGDM specific functions ####

def _get_mgdm_orientation(affine, mgdm):
    '''
    Transforms nibabel affine information into
    orientation and slice order that MGDM understands
    '''
    orientation = nb.aff2axcodes(affine)
    if orientation[-1] == "I" or orientation[-1] == "S":
        sliceorder = mgdm.AXIAL"
    elif orientation[-1] == "L" or orientation[-1] == "R":
        sliceorder = mgdm.SAGITTAL
    else:
        sliceorder = mgdm.CORONAL
    return orientation, sliceorder


        for aff_orient in aff_orientations:
            if aff_orient == "L":
                LR = mgdm.R2L
            elif aff_orient == "R":
                LR = mgdm.L2R
               # flipLR = True
            elif aff_orient == "A":
                AP = mgdm.P2A
                #flipAP = True
            elif aff_orient == "P":
                AP = mgdm.A2P
            elif aff_orient == "I":
                IS = mgdm.S2I
                #flipIS = True
            elif aff_orient == "S":
                IS = mgdm.I2S


def _get_mgdm_intensity_priors(atlas_file):
    """
    Returns a list of available as intensity priors
    in the MGDM atlas that you are using
    :param atlas_file:              atlas file
    :return: seg_contrast_names     list of names of contrasts that have
                                    intensity priors available
    """
    priors = []
    with open(atlas_file) as fp:
        for i, line in enumerate(fp):
            if "Structures:" in line:  # this is the beginning of the LUT
                lut_idx = i
                lut_rows = map(int, [line.split()[1]])[0]
            if "Intensity Prior:" in line:
                priors.append(line.split()[-1])
    return priors


def mgdm_segmentation(con1_files, con1_type,
                      con2_files=None, con2_type=None,
                      con3_files=None, con3_type=None,
                      con4_files=None, con4_type=None,
                      num_steps=5, topology='wcs',
                      atlas_file=None, topology_lut_dir=None,
                      adjust_intensity_priors=False,
                      compute_posterior=False,
                      diffuse_probabilities=False,
                      output_dir=None, file_suffix=None):
    """
    Performs MGDM segmentation
    simplified inputs for a total of 4 different contrasts
    :param con1_files:              List of files for contrast 1, required
    :param con1_type:               Contrast 1 type (from __get_mgdm_intensity_priors(atlas_file))
    :param con2_files:              List of files for contrast 2, optional, must be matched to con1_files
    :param con2_type:               Contrast 2 type
    :param con3_files:              List of files for contrast 3, optional, must be matched to con1_files
    :param con3_type:               Contrast 3 type
    :param con4_files:              List of files for contrast 4, optional, must be matched to con1_files
    :param con4_type:               Contrast 4 type
    :param output_dir:              Directory to place output, defaults to input directory if = None
    :param num_steps:               Number of steps for MGDM, default = 5, set to 0 for quick testing of registration of priors (not true segmentation)
    :param topology:                Topology setting {'wcs', 'no'} ('no' for no topology)
    :param atlas_file:              Path to atlas file
    :param topology_lut_dir:        Directory for topology files
    :param adjust_intensity_priors: Adjust intensity priors based on dataset: True/False
    :param compute_posterior:       Compute posterior: True/False
    :param diffuse_probabilities:   Compute diffuse probabilities: True/False
    :param file_suffix:             Distinguishing text to add to the end of the filename
    :return:
    """

    # find available intensity priors in selected MGDM atlas
    mgdm_intensity priors = __get_mgdm_intensity_priors(atlas_file)

    # TODO adpat error messages
    if con1_type not in mgdm_intensity priors:
        print("You have not chosen a valid contrast ({0}) for your metric_contrast_name, please choose from: ").format(
            con1_type)
        print(", ".join(mgdm_intensity priors))
        return
    if con2_type is not None and con2_type not in mgdm_intensity priors:
        print("You have not chosen a valid contrast ({0}) for your metric_contrast_name, please choose from: ").format(
            con2_type)
        print(", ".join(mgdm_intensity priors))
        return
    if con3_type is not None and con3_type not in mgdm_intensity priors:
        print("You have not chosen a valid contrast ({0}) for your metric_contrast_name, please choose from: ").format(
            con3_type)
        print(", ".join(mgdm_intensity priors))
        return
    if con4_type is not None and con4_type not in mgdm_intensity priors:
        print("You have not chosen a valid contrast ({0}) for your metric_contrast_name, please choose from: ").format(
            con4_type)
        print(", ".join(mgdm_intensity priors))
        return

    out_files_seg = []
    out_files_lbl = []
    out_files_ids = []

    # set output dir if not given, create output dir if not exist
    if output_dir is None:
        output_dir = os.path.dirname(con1_files[0])
    create_dir(output_dir)

    # set default atlas if not given
    # TODO search given atlas file in default atlas dir?
    if atlas_file is None:
        atlas = os.path.join(ATLAS_DIR, DEFAULT_ATLAS)
    else:
        atlas = atlas_file

    # set default topology lut dir if not given
    if topology_lut_dir is None:
        topology_lut_dir = TOPOLOGY_LUT_DIR
    else:
        # if we don't end in a path sep, we need to make sure that we add it
        if not(topology_lut_dir[-1] == os.path.sep):
            topology_lut_dir += os.path.sep

    print("Atlas file: " + atlas)
    print("Topology LUT directory: " + topology_lut_dir)
    print("")

    if not isinstance(con1_files, list):  # make into lists if they were not
        con1_files = [con1_files]
    if con2_files is not None and not isinstance(con2_files, list):
        con2_files = [con2_files]
    if con3_files is not None and not isinstance(con3_files, list):
        con3_files = [con3_files]
    if con4_files is not None and not isinstance(con4_files, list):
        con4_files = [con4_files]

    # set  mgdm specfic settings
    mgdm = cbstools.BrainMgdmMultiSegmentation2()
    mgdm.setAtlasFile(atlas)
    mgdm.setTopologyLUTdirectory(topology_lut_dir)

    mgdm.setOutputImages('segmentation')
    mgdm.setAdjustIntensityPriors(adjust_intensity_priors)   # default is False
    mgdm.setComputePosterior(compute_posterior)  # default is False
    mgdm.setDiffuseProbabilities(diffuse_probabilities)  # default is False
    mgdm.setSteps(num_steps)  # default is 5
    mgdm.setTopology(topology)  # {'wcs','no'} no=off for testing, wcs=default
    # --> mgdm.setOrientations(mgdm.AXIAL, mgdm.R2L, mgdm.A2P, mgdm.I2S) # this is the default of the atlas used for MGDM, <--

    for idx, con1 in enumerate(con1_files):
        print("Input files and filetypes:")
        print(con1_type + ":\t" + con1.split(os.path.sep)[-1])

        # load volume
        img = load_volume(con1)
        data = img.get_data()
        affine = img.get_affine()
        header = img.get_header()

        # convert orientation information to mgdm slice and orientation info
        orientation, sliceorder = _get_mgdm_orientation(affine)
        print("data orientation: " + str(orientation)),
        mgdm.setOrientations(sliceorder, LR, AP, IS)
        # L2R,P2A,I2S is nibabel default (i.e., RAS)

        # we use the first image to set the dimensions and resolutions
        res = d_head.get_zooms()
        res = [a1.item() for a1 in res]  # cast to regular python float type
        mgdm.setDimensions(d.shape[0], d.shape[1], d.shape[2])
        mgdm.setResolutions(res[0], res[1], res[2])

        # keep the shape and affine from the first image for saving
        d_shape = np.array(d.shape)
        out_root_fname = os.path.basename(fname)[0:os.path.basename(
            fname).find('.')]  # assumes no periods in filename, :-/
        mgdm.setContrastImage1(cj.JArray('float')((d.flatten('F')).astype(float)))
        mgdm.setContrastType1(con1_type)

        if con2_files is not None:  # only bother with the other contrasts if something is in the one before it
            print(con2_type + ":\t" + con2_files[idx].split(os.path.sep)[-1])
            d, a = niiLoad(con2_files[idx], return_header=False)
            mgdm.setContrastImage2(cj.JArray('float')((d.flatten('F')).astype(float)))
            mgdm.setContrastType2(con2_type)
            if con3_files is not None:
                print(con3_type + ":\t" + con3_files[idx].split(os.path.sep)[-1])
                d, a = niiLoad(con3_files[idx], return_header=False)
                mgdm.setContrastImage3(cj.JArray('float')((d.flatten('F')).astype(float)))
                mgdm.setContrastType3(con3_type)
                if con4_files is not None:
                    print(con4_type + ":\t" + con4_files[idx].split(os.path.sep)[-1])
                    d, a = niiLoad(con4_files[idx], return_header=False)
                    mgdm.setContrastImage4(cj.JArray('float')((d.flatten('F')).astype(float)))
                    mgdm.setContrastType4(con4_type)
        try:
            print("Executing MGDM on your inputs")
            print("Don't worry, the magic is happening!")
            ## ---------------------------- MGDM MAGIC START ---------------------------- ##
            mgdm.execute()
            ## ---------------------------- MGDM MAGIC END   ---------------------------- ##

            # outputs
            # reshape fortran stype to convert back to the format the nibabel likes
            seg_im = np.reshape(np.array(mgdm.getSegmentedBrainImage(),
                                         dtype=np.uint32), d_shape, 'F')
            lbl_im = np.reshape(np.array(mgdm.getPosteriorMaximumLabels4D(),
                                         dtype=np.uint32), d_shape, 'F')
            ids_im = np.reshape(np.array(mgdm.getSegmentedIdsImage(),
                                         dtype=np.uint32), d_shape, 'F')

            # filenames for saving
            if file_suffix is not None:
                seg_file = os.path.join(output_dir, out_root_fname +
                                        '_seg' + file_suffix + '.nii.gz')
                lbl_file = os.path.join(output_dir, out_root_fname +
                                        '_lbl' + file_suffix + '.nii.gz')
                ids_file = os.path.join(output_dir, out_root_fname +
                                        '_ids' + file_suffix + '.nii.gz')
            else:
                seg_file = os.path.join(output_dir, out_root_fname + '_seg.nii.gz')
                lbl_file = os.path.join(output_dir, out_root_fname + '_lbl.nii.gz')
                ids_file = os.path.join(output_dir, out_root_fname + '_ids.nii.gz')

            d_head['data_type'] = np.array(32).astype('uint32')  # convert the header as well
            d_head['cal_max'] = np.max(seg_im)  # max for display
            niiSave(seg_file, seg_im, d_aff, header=d_head, data_type='uint32')
            d_head['cal_max'] = np.max(lbl_im)
            niiSave(lbl_file, lbl_im, d_aff, header=d_head, data_type='uint32')
            d_head['cal_max'] = np.max(ids_im)  # convert the header as well
            niiSave(ids_file, ids_im, d_aff, header=d_head, data_type='uint32')
            print("Data stored in: " + output_dir)
            print("")
            out_files_seg.append(seg_file)
            out_files_lbl.append(lbl_file)
            out_files_ids.append(ids_file)
        except:
            print("--- MGDM failed. Go cry. ---")
            return
        print("Execution completed")

    return out_files_seg, out_files_lbl, out_files_ids


def compare_atlas_segs_priors(seg_file_orig, seg_file_new, atlas_file_orig=None, atlas_file_new=None,
                              metric_contrast_name=None, background_idx=1, seg_null_value=0):
    """
    Compare a new segmentation and atlas priors to another. Comparison is made relative to the orig
    :param seg_file_orig:
    :param atlas_file_orig:
    :param seg_file_new:
    :param atlas_file_new:
    :param metric_contrast_name:        Contrast type from atlas file
    :return:
    """
    import numpy as np

    d1, a1 = niiLoad(seg_file_orig, return_header=False)
    d2, a2 = niiLoad(seg_file_new, return_header=False)
    idxs1 = np.unique(d1)
    idxs2 = np.unique(d2)
    [lut1, con_idx1, lut_rows1, priors1] = extract_lut_priors_from_atlas(
        atlas_file_orig, metric_contrast_name)
    # TODO: make sure that all indices are in both segs? or just base it all on the gold standard?

    for struc_idx in lut1.Index:
        if not(struc_idx == background_idx):
            print("Structure index: {0}, {1}").format(
                struc_idx, lut1.index[lut1.Index == struc_idx][0])
            bin_vol = np.zeros_like(d1)
            bin_vol[d1 == struc_idx] = 1
            dice = np.sum(bin_vol[d2 == struc_idx]) * 2.0 / \
                (np.sum(bin_vol) + np.sum(d2 == struc_idx))
            print("Dice similarity: {}").format(dice)
            # identify misclassifications
            bin_vol = np.ones_like(d1) * seg_null_value
            bin_vol[d1 == struc_idx] = 1

            overlap = np.multiply(bin_vol, d2)
            overlap_idxs = np.unique(overlap)
            # remove the idx that we should be at the moment
            overlap_idxs = np.delete(overlap_idxs, np.where(overlap_idxs == struc_idx))
            # remove the null value, now left with the overlap with things we don't want :-(
            overlap_idxs = np.delete(overlap_idxs, np.where(overlap_idxs == seg_null_value))
            # print overlap_idxs

    # TODO: overlap comparison here

    #[lut2, con_idx2, lut_rows2, priors2] = extract_lut_priors_from_atlas(atlas_file_new, metric_contrast_name)
    # TODO: based on overlap comparison, adjust intensity priors
    return lut1


def seg_erode(seg_d, iterations=1, background_idx=1,
              structure=None, min_vox_count=5, seg_null_value=0,
              VERBOSE=False):
    """
    Binary erosion (or dilation) of integer type segmentation data (np.array) with options
    If iterations < 0, performs binary dilation
    :param seg_d:           np.array of segmentation, integers
    :param iterations:      number of erosion iterations, if negative, provides the number of dilations (in this case, min_vox_count not used)
    :param background_idx:  value for background index, currently ignored (TODO: remove)
    :param structure:       binary structure for erosion from scipy.ndimage (ndimage.morphology.generate_binary_structure(3,1))
    :param min_vox_count:   minimun number of voxels to allow to be in a segmentation, if less, does not erode
    :param seg_null_value:  value to set as null for binary erosion step (i.e., a value NOT in your segmentation index)
    :param VERBOSE:         spit out loads of text to stdout, because you can.
    :return: seg_shrunk_d   eroded (or dilated) version of segmentation
    """

    import scipy.ndimage as ndi
    import numpy as np

    if iterations >= 0:
        pos_iter = True
    else:
        iterations = iterations * -1
        pos_iter = False

    if structure is None:
        structure = ndi.morphology.generate_binary_structure(3, 1)
    if seg_null_value == 0:
        seg_shrunk_d = np.zeros_like(seg_d)
        temp_d = np.zeros_like(seg_d)
    else:
        seg_shrunk_d = np.ones_like(seg_d) * seg_null_value
        temp_d = np.ones_like(seg_d) * seg_null_value

    seg_idxs = np.unique(seg_d)

    if seg_null_value in seg_idxs:
        print("Shit, your null value is also an index. This will not work.")
        print("Set it to a suitably strange value that is not already an index. {0,999}")
        return None
    if VERBOSE:
        print("Indices:")
    for seg_idx in seg_idxs:
        if VERBOSE:
            print(seg_idx),
        if (background_idx is not None) and (background_idx == seg_idx):
            # just set the value to the bckgrnd value, and be done with it
            seg_shrunk_d[seg_d == seg_idx] = seg_idx
            if VERBOSE:
                print("[bckg]"),
        else:
            temp_d[seg_d == seg_idx] = 1
            # messy, does not exit the loop when already gone too far. but it still works
            for idx in range(0, iterations):
                if pos_iter:
                    temp_temp_d = ndi.binary_erosion(temp_d, iterations=1, structure=structure)
                else:
                    temp_temp_d = ndi.binary_dilation(temp_d, iterations=1, structure=structure)
                if np.sum(temp_temp_d) >= min_vox_count:
                    temp_d = temp_temp_d
                    if VERBOSE:
                        print("[y]"),
                else:
                    if VERBOSE:
                        print("[no]"),
            seg_shrunk_d[temp_d == 1] = seg_idx
            temp_d[:, :, :] = seg_null_value
            if VERBOSE:
                print(seg_idx)
        if VERBOSE:
            print("")
    return seg_shrunk_d


def extract_metrics_from_seg(seg_d, metric_d, seg_idxs=None, norm_data=True,
                             background_idx=1, seg_null_value=0,
                             percentile_top_bot=[75, 25],
                             return_normed_metric_d=False):
    """
    Extract median and interquartile range from metric file given a co-registered segmentation
    :param seg_d:                   segmentation data (integers)
    :param metric_d:                metric data to extract seg-specific values from
    :param seg_idxs:                indices of segmentation, usually taken from LUT but can be generated based on seg_d
    :param norm_data:               perform data normalisation on metric_d prior to extracting values from metric
    :param background_idx:          index for background data, currently treated as just another index (TODO: remove)
    :param seg_null_value:          value to set as null for binary erosion step, not included in metric extraction
    :param percentile_top_bot:      top and bottom percentiles to extract from each seg region
    :param return_normed_metric_d:  return the normalised metric as an np matrix, must also set norm_data=True
    :return: seg_idxs, res          segmentation indices and results matrix of median, 75, 25 percentliles
             (metric_d)             optional metric_d scaled between 0 and 1
    """
    import numpy as np
    if seg_idxs is None:
        seg_idxs = np.unique(seg_d)
    # remove the null value from the idxs so we don't look
    if (seg_null_value is not None) and (seg_null_value in seg_idxs):
        np.delete(seg_idxs, np.where(seg_idxs == seg_null_value))
    res = np.zeros((len(seg_idxs), 3))

    if norm_data:  # rescale the data to 0
        if background_idx is not None:  # we need to exclude the background data from the norming
            metric_d[seg_d != background_idx] = (metric_d[seg_d != background_idx] - np.min(
                metric_d[seg_d != background_idx])) / (np.max(metric_d[seg_d != background_idx]) - np.min(
                    metric_d[seg_d != background_idx]))
        else:
            metric_d = (metric_d - np.min(metric_d)) / (np.max(metric_d) - np.min(metric_d))

    for idx, seg_idx in enumerate(seg_idxs):
        d_1d = np.ndarray.flatten(metric_d[seg_d == seg_idx])
        res[idx, :] = [np.median(d_1d),
                       np.percentile(d_1d, np.max(percentile_top_bot)),
                       np.percentile(d_1d, np.min(percentile_top_bot))]
    if return_normed_metric_d:
        return seg_idxs, res, metric_d
    else:
        return seg_idxs, res


def extract_lut_priors_from_atlas(atlas_file, contrast_name):
    """
    Given an MGDM segmentation priors atlas file, extract the lut and identify the start index (in the file) of the
    contrast of interest, and the number of rows of priors that it should have. Returns pandas dataframe of lut,
    contrast index, number of rows in prior definition, and pd.DataFrame of priors,
    :param atlas_file:      full path to atlas file for lut and metric index extraction
    :param contrast_name:   intensity prior contrast name as listed in the metric file
    :return: lut, con_idx, lut_rows, priors
    """
    import pandas as pd

    fp = open(atlas_file)
    for i, line in enumerate(fp):
        if "Structures:" in line:  # this is the beginning of the LUT, grab the number of items in the atlas from this line
            lut_idx = i
            # g+1 to ensure that the last line is included
            lut_rows = map(int, [line.split()[1]])[0] + 1
        if "Intensity Prior:" in line:
            if contrast_name in line:
                con_idx = i
    fp.close()

    # dump lut and priors values into pandas dataframes
    lut = pd.read_csv(atlas_file, sep="\t+",
                      skiprows=lut_idx + 1, nrows=lut_rows, engine='python',
                      names=["Index", "Type"])

    priors = pd.read_csv(atlas_file, sep="\t+",
                         skiprows=con_idx + 1, nrows=lut_rows, engine='python',
                         names=["Median", "Spread", "Weight"])
    return lut, con_idx, lut_rows, priors


def write_priors_to_atlas(prior_medians, prior_quart_diffs, atlas_file, new_atlas_file, metric_contrast_name):
    """
    Write modified priors of given metric contrast to new_atlas
    Assumes that the ordering of indices and the ordering of the priors are the same
    (could add prior_weights as well, in future, and use something more structured than just line reading and writing)
    adds the additional contrast to the end of the file if it does not already exist
    :param prior_medians:           2xN list of prior medians
    :param prior_quart_diffs:       2xN list of prior quartile differences
    :param atlas_file:              full path to original atlas file
    :param new_atlas_file:          full path to new atlas file to be written to
    :param metric_contrast_name:    name of MGDM metric contrast from atlas_file
    :return: fp_new.name            name of newly written atlas file
    """

    import pandas as pd
    contrast_names = __get_mgdm_intensity_priors(atlas_file)
    if metric_contrast_name not in contrast_names:
        # then we need to append this to the end of the file rather than reset the values that existed
        NEW_CONTRAST = True

    if not NEW_CONTRAST:
        # get the relevant information from the old atlas file
        [lut, con_idx, lut_rows, priors] = extract_lut_priors_from_atlas(
            atlas_file, metric_contrast_name)
        seg_idxs = lut.Index.get_values()  # np vector of index values
        priors_new = pd.DataFrame.copy(priors)

        # uppdate the priors with the new ones that were passed
        # TODO: double-check this
        for idx in lut.Index:
            priors_new[lut["Index"] == idx] = [
                prior_medians[seg_idxs == idx], prior_quart_diffs[seg_idxs == idx], 1]

        priors_new_string = priors_new.to_csv(sep="\t", header=False, float_format="%.2f")
        # convert to list of lines, cut the last empty '' line
        priors_new_string_lines = priors_new_string.split("\n")[0:-1]

        fp = open(atlas_file)
        fp_new = open(new_atlas_file, "w")
        ii = 0
        # only replace the lines that we changed
        for i, line in enumerate(fp):
            if i > con_idx and i < con_idx + lut_rows:
                fp_new.write(priors_new_string_lines[ii] + "\n")
                ii += 1
            else:
                fp_new.write(line)
        fp.close()
        fp_new.close()
    else:  # this is a new contrast, so get the original atlas_file and then append our new priors to it
        [lut, con_idx, lut_rows, priors] = extract_lut_priors_from_atlas(
            atlas_file, contrast_names[0])
        seg_idxs = lut.Index.get_values()  # np vector of index values
        priors_new = pd.DataFrame.copy(priors)
        with open(atlas_file) as f:  # copy the file to a list so that we can append to it
            content = f.readlines()
        if content[-1] is not ' \n':  # the space is in the original atlas file, so we keep it
            content.append(' \n')
        for idx in lut.Index:
            priors_new[lut["Index"] == idx] = [
                prior_medians[seg_idxs == idx], prior_quart_diffs[seg_idxs == idx], 1]
        header = "Intensity Prior:\t" + metric_contrast_name + '\n'
        priors_new_string = priors_new.to_csv(sep="\t", header=False, float_format="%.2f")
        # convert to list of lines, cut the last empty '' line
        priors_new_string_lines = priors_new_string.split("\n")[0:-1]
        priors_new_string_lines.insert(0, header)
        priors_new_string_lines = [
            theLine + '\n' for theLine in priors_new_string_lines]  # append the \n back on
        content = content + priors_new_string_lines
        fp_new = open(new_atlas_file, "w")
        for line in content:
            fp_new.write(line)
        fp_new.close()
    print('New atlas file written to: \n' + fp_new.name)
    return fp_new.name


def filter_sigmoid(d, x0=0.002, slope=0.0005, output_fname=None):
    """
    Pass data through a sigmoid filter (scaled between 0 and 1). Defaults set for MD rescaling
    If you are lazy and pass it a filename, it will pass you back the data with affine and header
    :param d:
    :param x0:
    :param slope:
    :return:
    """
    import numpy as np

    return_nii_parts = False
    if not isinstance(d, (np.ndarray, np.generic)):
        try:
            [d, a, h] = niiLoad(d, return_header=True)
            return_nii_parts = True
        except:
            print("niiLoad tried to load this is a file and failed, are you calling it properly?")
            return
    if output_fname is not None and return_nii_parts:
        niiSave(output_fname, d, a, h)
    if return_nii_parts:
        return 1 / (1 + np.exp(-1 * (d - x0) / slope)), a, h
    else:
        return 1 / (1 + np.exp(-1 * (d - x0) / slope))



def niiSave(nii_fname, d, affine, header=None, data_type=None):
    """
    Save nifti image to file
    :param nii_fname:
    :param d:
    :param affine:
    :param header:      text of numpy data_type (e.g. 'uint32','float32')
    :param data_type:
    :return: nii_fname: filename that was written to disk
    """
    import nibabel as nb

    if data_type is not None:
        d.astype(data_type)
    img = nb.Nifti1Image(d, affine, header=header)
    if data_type is not None:
        img.set_data_dtype(data_type)
    img.to_filename(nii_fname)
    return nii_fname








def generate_group_intensity_priors(orig_seg_files, metric_files, orig_metric_contrast_name,
                                    atlas_file, erosion_iterations=1, min_quart_diff=0.1,
                                    seg_null_value=0, background_idx=1,
                                    VERBOSE=False, intermediate_output_dir=None):
    """
    generates group intensity priors for metric_files based on orig_seg files (i.e., orig_seg could be Mprage3T and metric_files could be DWIFA3T)
    does not do the initial segmentation for you, that needs to be done first :-)
    we assume that you already did due-diligence and have matched lists of inputs (orig_seg_files and metric_files)
    :param orig_seg_files:          segmentation from other modality
    :param metric_files:            metric files in same space as orig_seg_files
    :param orig_metric_contrast_name:    name of contrast from priors atlas file, not used currently
    :param atlas_file:              prior atlas file (use os.path.join(ATLAS_DIR,DEFAULT_ATLAS))
    :param erosion_iterations:      number of voxels to erode from each segmented region prior to metric extraction
    :param min_quart_diff:          minimum difference between quartiles to accept, otherwise replace with this
    :param seg_null_value:          null value for segmentation results (choose a value that is not in your seg, usually 0)
    :param background_idx:          background index value (usually 1, to leave 0 as a seg_null_value)
    :param VERBOSE:
    :return: medians, spread        metric-specific prior medians and spread for atlas file
    """
    import nibabel as nb
    import numpy as np
    import os

    mgdm_intensity priors = __get_mgdm_intensity_priors(atlas_file)
    if orig_metric_contrast_name not in mgdm_intensity priors:
        print("You have not chosen a valid contrast for your metric_contrast_name, please choose from: ")
        print(", ".join(mgdm_intensity priors))
        return [None, None]

    [lut, con_idx, lut_rows, priors] = extract_lut_priors_from_atlas(
        atlas_file, orig_metric_contrast_name)
    seg_idxs = lut.Index
    all_Ss_priors_median = np.array(seg_idxs)  # always put the seg_idxs on top row!
    all_Ss_priors_spread = np.array(seg_idxs)
    # seg_null_value = 0 #value to fill in when we are NOT using the voxels at all (not background and not other index)
    #background_idx = 1
    # min_quart_diff = 0.10 #minimun spread allowed in priors atlas

    # make a list if we only input one dataset
    if len(orig_seg_files) == 1:
        orig_seg_files = [orig_seg_files]
    if len(metric_files) == 1:
        metric_files = [metric_files]

    if not(len(orig_seg_files) == len(metric_files)):
        print("You do not have the same number of segmentation and metric files. Bad!")
        print("Exiting")
        return [None, None]
    if erosion_iterations > 0:
        print("Performing segmentation erosion on each segmented region with %i step(s)" %
              erosion_iterations)

    for idx, seg_file in enumerate(orig_seg_files):
        metric_file = metric_files[idx]
        img = nb.load(metric_file)
        d_metric = img.get_data()
        a_metric = img.affine  # not currently using the affine and header, but could also output the successive steps
        h_metric = img.header
        print(seg_file.split(os.path.sep)[-1])
        print(metric_file.split(os.path.sep)[-1])
        d_seg = nb.load(seg_file).get_data()

        # erode our data
        if erosion_iterations > 0:
            d_seg_ero = seg_erode(d_seg, iterations=erosion_iterations,
                                  background_idx=background_idx,
                                  seg_null_value=seg_null_value)
        else:
            d_seg_ero = d_seg

        # extract summary metrics (median, 75 and 25 percentile) from metric file
        [seg_idxs, seg_stats] = extract_metrics_from_seg(d_seg_ero, d_metric, seg_idxs=seg_idxs,
                                                         seg_null_value=seg_null_value,
                                                         return_normed_metric_d=False)

        prior_medians = seg_stats[:, 0]
        prior_quart_diffs = np.squeeze(np.abs(np.diff(seg_stats[:, 1:3])))
        prior_quart_diffs[prior_quart_diffs < min_quart_diff] = min_quart_diff

        # now place this output into a growing array for use on the group level
        all_Ss_priors_median = np.vstack((all_Ss_priors_median, prior_medians))
        all_Ss_priors_spread = np.vstack((all_Ss_priors_spread, prior_quart_diffs))

        if intermediate_output_dir is not None:
            img = nb.Nifti1Image(d_seg_ero, a_metric, header=h_metric)
            img.to_filename(os.path.join(intermediate_output_dir, seg_file.split(
                os.path.sep)[-1].split(".")[0] + "_ero" + str(erosion_iterations) + ".nii.gz"))
        print("")
    return all_Ss_priors_median, all_Ss_priors_spread


def limit_to_robust_range(d, min_value=None, max_value=None, lower_percentile=0.1, upper_percentile=99.9):
    """
    Limits input data to robust range as specified by min_value, lower_percentile, and upper_percentile.
    May be useful after bias correction or noise correction to ensure that the histogram is not weighted by erroneous signal
    :param d:                   input data
    :param min_value:           values less than this will be set to this; if None, not considered
    :param lower_percentile:    values less than this percentile from data will be set to this
    :param upper_percentile:    values greater than this percentile from data will be set to this
    :return: limited data
    """
    if min_value is not None:
        d[d < min_value] = min_value
    if max_value is not None:
        d[d > max_value] = max_value
    minimum = np.percentile(d, lower_percentile)
    d[d < minimum] = minimum
    maximum = np.percentile(d, upper_percentile)
    d[d > maximum] = maximum
    print ("lower: {0}\t upper: {1}").format(minimum, maximum)
    return d


def iteratively_generate_group_intensity_priors(con1_files, con1_type, orig_seg_files, orig_metric_contrast_name, atlas_file, con2_files=None,
                                                con2_type=None, con3_files=None, con3_type=None, con4_files=None,
                                                con4_type=None, output_dir=None, num_steps=5, topology='wcs',
                                                topology_lut_dir=None, adjust_intensity_priors=False,
                                                compute_posterior=False, diffuse_probabilities=False,
                                                file_suffix=None, new_atlas_file_head=None, make_new_contrast=False,
                                                erosion_iterations=1, seg_iterations=1):
    # from multiprocessing import pool
    # mgdm_pool = pool.Pool(processes = num_parallel_jobs)
    # mgdm_pool.map(COMMAND, INPUT)
    import numpy as np
    import os
    current_atlas_file = atlas_file
    if new_atlas_file_head is None:
        # we cut off the .txt, and add our mod txt, we don't check if it already exists
        new_atlas_file_head = atlas_file.split('.txt')[0] + "_mod"
    if not isinstance(con1_files, list):  # make into lists if they were not
        con1_files = [con1_files]
    if con2_files is not None and not isinstance(con2_files, list):  # make into list
        con2_files = [con2_files]
    if con3_files is not None and not isinstance(con3_files, list):  # make into list
        con3_files = [con3_files]
    if con4_files is not None and not isinstance(con4_files, list):  # make into list
        con4_files = [con4_files]

    mgdm_intensity priors = __get_mgdm_intensity_priors(
        atlas_file)  # get contrast names from old atlas file
    contrast_names = __get_mgdm_intensity_priors(atlas_file)
    if make_new_contrast:
        # list that tells us if we have new contrasts or not, so that we can loop over them later if we need to write them to file
        new_contrasts = [False, False, False, False]
        if con1_type not in contrast_names:
            new_contrasts[0] = True
        if con2_type not in contrast_names:
            new_contrasts[1] = True
        if con3_type not in contrast_names:
            new_contrasts[2] = True
        if con4_type not in contrast_names:
            new_contrasts[3] = True

    # index of valid contrasts (this could be included in more complex statement above, but here for clarity
    contrast_list = [con1_type, None, None, None]
    if con2_type is not None:
        contrast_list[1] = con2_type
    if con3_type is not None:
        contrast_list[2] = con3_type
    if con4_type is not None:
        contrast_list[3] = con4_type

    # the first time, we just grab the metric data and update the priors atlas
    seg_iter_text = str(0).zfill(3)  # text for naming files etc
    print("First pass with no segmentation: " + seg_iter_text)
    print("Calculating priors from input metric files.")

    # first pass at generating the intensity priors for the input contrasts
    for idx, val in enumerate(contrast_list):  # need to loop extractions and priors updating over metrics
        if val is not None:
            metric_contrast_name = val
        else:
            return  # do something here to skip this loop?

        print("Metric type: " + metric_contrast_name)
        if idx + 1 == 1:
            metric_files = con1_files
        elif idx + 1 == 2:
            metric_files = con2_files
        elif idx + 1 == 3:
            metric_files = con3_files
        elif idx + 1 == 4:
            metric_files = con4_files

        # new atlas file name changes with iteration AND with metric name, to make sure that we keep track of everything
        new_atlas_file = os.path.join(new_atlas_file_head + "_" +
                                      seg_iter_text + "_" + metric_contrast_name + ".txt")
        [priors_median, priors_spread] = generate_group_intensity_priors(orig_seg_files, metric_files,
                                                                         orig_metric_contrast_name,
                                                                         current_atlas_file,
                                                                         erosion_iterations=erosion_iterations,
                                                                         output_dir=output_dir)
        seg_idxs = priors_median[0, :]
        grp_median = np.median(priors_median[1:, :], axis=0)
        grp_spread = np.median(priors_spread[1:, :], axis=0)
        write_priors_to_atlas(grp_median, grp_spread, current_atlas_file,
                              new_atlas_file, metric_contrast_name)
        # update the current atlas file, so that we can use it for subsequent extractions
        current_atlas_file = new_atlas_file

        # combine the individual output into a 2d and then 3d stack (with iterations >0) so that we can keep track of changes
        # it will be stacked for each metric if there are multiple metrics, so not easy to see :-/
        iter_Ss_priors_median = priors_median
        iter_Ss_priors_spread = priors_spread

    # now run the segmentation for each individual with this current_atlas_file
    #new_seg_files = []
    for seg_iter in range(0, seg_iterations):
        seg_iter_text = str(seg_iter + 1).zfill(3)  # text for naming files etc
        print("Running segmentation iteration: " + seg_iter_text)
        new_seg_files = []  # list to contain the output segmentation files from the current step

        # for idx,con1_file in enumerate(con1_files):
        #     con2_file = None
        #     con3_file = None
        #     con4_file = None
        #
        #     if con2_type is not None:
        #         con2_file = con2_files[idx]
        #     if con3_type is not None:
        #         con3_file = con3_files[idx]
        #     if con4_type is not None:
        #         con4_file = con4_files[idx]

        # RUN SEGMENTATION with current atlas file
        # TODO: stupid parallelisation?
        # could use the code above to submit multiple jobs with individuals
        MGDM_output_files = MGDMBrainSegmentation(con1_files, con1_type, con2_files=con2_files, con2_type=con2_type,
                                                  con3_files=con3_files, con3_type=con3_type, con4_files=con4_files, con4_type=con4_type,
                                                  output_dir=output_dir, num_steps=num_steps, topology=topology, atlas_file=current_atlas_file,
                                                  topology_lut_dir=topology_lut_dir, adjust_intensity_priors=adjust_intensity_priors, compute_posterior=compute_posterior,
                                                  diffuse_probabilities=diffuse_probabilities, file_suffix=file_suffix)
        new_seg_files = MGDM_output_files[0]  # segmentations are passed first

        # RUN EXTRACTION FOR EACH METRIC on output from segmentation, UPDATE atlas priors
        print("Metric extraction from new segmentation")

        # need to loop extractions and priors, updating over metrics
        for idx, val in enumerate(contrast_list):
            if val is not None:
                metric_contrast_name = val
            else:
                return  # do something here to skip this loop?

            print("Metric type: " + metric_contrast_name)
            if idx + 1 == 1:
                metric_files = con1_files
            elif idx + 1 == 2:
                metric_files = con2_files
            elif idx + 1 == 3:
                metric_files = con3_files
            elif idx + 1 == 4:
                metric_files = con4_files

            # new atlas file name changes with iteration AND with metric name, to make sure that we keep track of everything
            new_atlas_file = os.path.join(new_atlas_file_head + "_" +
                                          seg_iter_text + "_" + metric_contrast_name + ".txt")
            [priors_median, priors_spread] = generate_group_intensity_priors(new_seg_files, metric_files,
                                                                             orig_metric_contrast_name,
                                                                             current_atlas_file,
                                                                             erosion_iterations=erosion_iterations,
                                                                             output_dir=output_dir)
            seg_idxs = priors_median[0, :]
            grp_median = np.median(priors_median[1:, :], axis=0)
            grp_spread = np.median(priors_spread[1:, :], axis=0)
            write_priors_to_atlas(grp_median, grp_spread, current_atlas_file,
                                  new_atlas_file, metric_contrast_name)
            # update the current atlas file, so that we can use it for subsequent extractions
            current_atlas_file = new_atlas_file

            # combine the individual output into a 2d and then 3d stack (with iterations >0) so that we can keep track of changes
            # it will be stacked for each metric if there are multiple metrics, so not easy to see :-/
            iter_Ss_priors_median = priors_median
            iter_Ss_priors_spread = priors_spread


return current_atlas_file
