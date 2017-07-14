import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving
import pdb

# TODO
ATLAS_DIR = '/home/julia/workspace/cbstools-python/atlases/brain-segmentation-prior3.0/'
TOPOLOGY_LUT_DIR = '/home/julia/workspace/cbstools-python/lut/'
DEFAULT_ATLAS = "brain-atlas-3.0.3.txt"


def _get_mgdm_orientation(affine, mgdm):
    '''
    Transforms nibabel affine information into
    orientation and slice order that MGDM understands
    '''
    orientation = nb.aff2axcodes(affine)
    # set mgdm slice order
    # TODO how clean is this?
    if orientation[-1] == "I" or orientation[-1] == "S":
        sliceorder = mgdm.AXIAL
    elif orientation[-1] == "L" or orientation[-1] == "R":
        sliceorder = mgdm.SAGITTAL
    else:
        sliceorder = mgdm.CORONAL

    # set mgdm orientations
    if "L" in orientation:
        LR = mgdm.R2L
    elif "R" in orientation:
        LR = mgdm.L2R  # flipLR = True
    if "A" in orientation:
        AP = mgdm.P2A  # flipAP = True
    elif "P" in orientation:
        AP = mgdm.A2P
    if "I" in orientation:
        IS = mgdm.S2I  # flipIS = True
    elif "S" in orientation:
        IS = mgdm.I2S

    return sliceorder, LR, AP, IS


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


def mgdm_segmentation(contrast_image1, contrast_type1,
                      contrast_image2=None, contrast_type2=None,
                      contrast_image3=None, contrast_type3=None,
                      contrast_image4=None, contrast_type4=None,
                      n_steps=5, topology='wcs',
                      atlas_file=None, topology_lut_dir=None,
                      adjust_intensity_priors=False,
                      compute_posterior=False,
                      diffuse_probabilities=False,
                      save_data=False, output_dir=None,
                      file_name=None, file_extension=None):
    """
    Performs MGDM segmentation
    simplified inputs for a total of 4 different contrasts
    :param contrast_image1:              List of files for contrast 1, required
    :param contrast_type1:               Contrast 1 type (from __get_mgdm_intensity_priors(atlas_file))
    :param contrast_image2:              List of files for contrast 2, optional, must be matched to contrast_image1
    :param contrast_type2:               Contrast 2 type
    :param contrast_image3:              List of files for contrast 3, optional, must be matched to contrast_image1
    :param contrast_type3:               Contrast 3 type
    :param contrast_image4:              List of files for contrast 4, optional, must be matched to contrast_image1
    :param contrast_type4:               Contrast 4 type
    :param output_dir:              Directory to place output, defaults to input directory if = None
    :param n_steps:               Number of steps for MGDM, default = 5, set to 0 for quick testing of registration of priors (not true segmentation)
    :param topology:                Topology setting {'wcs', 'no'} ('no' for no topology)
    :param atlas_file:              Path to atlas file
    :param topology_lut_dir:        Directory for topology files
    :param adjust_intensity_priors: Adjust intensity priors based on dataset: True/False
    :param compute_posterior:       Compute posterior: True/False
    :param diffuse_probabilities:   Compute diffuse probabilities: True/False
    :return:
    """

    # set default atlas if not given
    # TODO search given atlas file in default atlas dir?
    if atlas_file is None:
        atlas_file = os.path.join(ATLAS_DIR, DEFAULT_ATLAS)

    # set default topology lut dir if not given
    if topology_lut_dir is None:
        topology_lut_dir = TOPOLOGY_LUT_DIR
    else:
        # if we don't end in a path sep, we need to make sure that we add it
        if not(topology_lut_dir[-1] == os.path.sep):
            topology_lut_dir += os.path.sep

    print("Atlas file: " + atlas_file)
    print("Topology LUT directory: " + topology_lut_dir)
    print("")

    # find available intensity priors in selected MGDM atlas
    mgdm_intensity_priors = _get_mgdm_intensity_priors(atlas_file)

    # sanity check contrast types
    for idx, ctype in enumerate([contrast_type1, contrast_type2,
                                 contrast_type3, contrast_type4]):
        if ctype is not None and ctype not in mgdm_intensity_priors:
            raise ValueError(("{0} is not a valid contrast type for  "
                              "contrast_type{1} please choose from the "
                              "following contrasts provided by the chosen "
                              "atlas: ").format(ctype, idx),
                             ", ".join(mgdm_intensity_priors))

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass
    # create mgdm instance
    mgdm = cbstools.BrainMgdmMultiSegmentation2()

    # set mgdm parameters
    mgdm.setAtlasFile(atlas_file)
    mgdm.setTopologyLUTdirectory(topology_lut_dir)
    mgdm.setOutputImages('segmentation')
    mgdm.setAdjustIntensityPriors(adjust_intensity_priors)
    mgdm.setComputePosterior(compute_posterior)
    mgdm.setDiffuseProbabilities(diffuse_probabilities)
    mgdm.setSteps(n_steps)
    mgdm.setTopology(topology)

    # load contrast image 1 and use it to set dimensions and resolution
    img = load_volume(contrast_image1)
    data = img.get_data()
    affine = img.get_affine()
    header = img.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    mgdm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    mgdm.setResolutions(resolution[0], resolution[1], resolution[2])

    # convert orientation information to mgdm slice and orientation info
    sliceorder, LR, AP, IS = _get_mgdm_orientation(affine, mgdm)
    mgdm.setOrientations(sliceorder, LR, AP, IS)

    # input image 1
    mgdm.setContrastImage1(cbstools.JArray('float')((data.flatten('F')).astype(float)))
    mgdm.setContrastType1(contrast_type1)

    # if further contrast are specified, input them
    if contrast_image2 is not None:
        data = load_volume(contrast_image2[idx]).get_data()
        mgdm.setContrastImage2(cbstools.JArray('float')((data.flatten('F')).astype(float)))
        mgdm.setContrastType2(contrast_type2)

        if contrast_image3 is not None:
            data = load_volume(contrast_image3[idx]).get_data()
            mgdm.setContrastImage3(cbstools.JArray('float')((data.flatten('F')).astype(float)))
            mgdm.setContrastType3(contrast_type3)

            if contrast_image4 is not None:
                data = load_volume(contrast_image4[idx]).get_data()
                mgdm.setContrastImage4(cbstools.JArray('float')((data.flatten('F')).astype(float)))
                mgdm.setContrastType4(contrast_type4)

    # execute MGDM
    try:
        print("Executing MGDM on your inputs")
        mgdm.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print ""
        print "The underlying Java code did not execute cleanly: "
        print sys.exc_info()[0]
        raise
        return

    # TODO collect other outputs
    # reshape output to what nibabel likes
    seg_data = np.reshape(np.array(mgdm.getSegmentedBrainImage(),
                                   dtype=np.uint32), dimensions, 'F')
    lbl_data = np.reshape(np.array(mgdm.getPosteriorMaximumLabels4D(),
                                   dtype=np.uint32), dimensions, 'F')
    ids_data = np.reshape(np.array(mgdm.getSegmentedIdsImage(),
                                    dtype=np.uint32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['data_type'] = np.array(32).astype('uint32')
    header['cal_max'] = np.max(seg_data)
    seg_nii = nb.Nifti1Image(seg_data, affine, header)

    header['cal_max'] = np.max(lbl_data)
    lbl_nii = nb.Nifti1Image(lbl_data, affine, header)

    header['cal_max'] = np.max(ids_data)
    ids_nii = nb.Nifti1Image(ids_data, affine, header)

    if save_data:
        output_dir = _output_dir_4saving(output_dir, contrast_image1)

        # TODO fix the suffixes
        seg_file = _fname_4saving(rootfile=contrast_image1,
                                  suffix='seg',
                                  base_name=file_name,
                                  extension=file_extension)

        lbl_file = _fname_4saving(rootfile=contrast_image1,
                                  suffix='lbls',
                                  base_name=file_name,
                                  extension=file_extension)

        ids_file = _fname_4saving(rootfile=contrast_image1,
                                  suffix='ids',
                                  base_name=file_name,
                                  extension=file_extension)

        save_volume(os.path.join(output_dir, seg_file), seg_nii)
        save_volume(os.path.join(output_dir, lbl_file), lbl_nii)
        save_volume(os.path.join(output_dir, ids_file), ids_nii)

    return seg_nii, lbl_nii, ids_nii
