import numpy as np
import nibabel as nb
import os
import cbstools
from ..io import load_volume, save_volume
from ..utils import create_dir

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


def mgdm_segmentation(con1_files, con1_type,
                      con2_files=None, con2_type=None,
                      con3_files=None, con3_type=None,
                      con4_files=None, con4_type=None,
                      num_steps=5, topology='wcs',
                      atlas_file=None, topology_lut_dir=None,
                      adjust_intensity_priors=False,
                      compute_posterior=False,
                      diffuse_probabilities=False,
                      save_data=False,
                      base_name=None, file_suffix=None):
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

    out_files_seg = []
    out_files_lbl = []
    out_files_ids = []

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

    # TODO adpat error messages
    if con1_type not in mgdm_intensity_priors:
        print("You have not chosen a valid contrast ({0}) for your metric_contrast_name, please choose from: ").format(
            con1_type)
        print(", ".join(mgdm_intensity_priors))
        return
    if con2_type is not None and con2_type not in mgdm_intensity_priors:
        print("You have not chosen a valid contrast ({0}) for your metric_contrast_name, please choose from: ").format(
            con2_type)
        print(", ".join(mgdm_intensity_priors))
        return
    if con3_type is not None and con3_type not in mgdm_intensity_priors:
        print("You have not chosen a valid contrast ({0}) for your metric_contrast_name, please choose from: ").format(
            con3_type)
        print(", ".join(mgdm_intensity_priors))
        return
    if con4_type is not None and con4_type not in mgdm_intensity_priors:
        print("You have not chosen a valid contrast ({0}) for your metric_contrast_name, please choose from: ").format(
            con4_type)
        print(", ".join(mgdm_intensity_priors))
        return

    if not isinstance(con1_files, list):  # make into lists if they were not
        con1_files = [con1_files]
    if con2_files is not None and not isinstance(con2_files, list):
        con2_files = [con2_files]
    if con3_files is not None and not isinstance(con3_files, list):
        con3_files = [con3_files]
    if con4_files is not None and not isinstance(con4_files, list):
        con4_files = [con4_files]

    # start virtual machine and create mgdm instance
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass
    mgdm = cbstools.BrainMgdmMultiSegmentation2()

    # set mgdm parameters
    mgdm.setAtlasFile(atlas_file )
    mgdm.setTopologyLUTdirectory(topology_lut_dir)
    mgdm.setOutputImages('segmentation')
    mgdm.setAdjustIntensityPriors(adjust_intensity_priors)
    mgdm.setComputePosterior(compute_posterior)
    mgdm.setDiffuseProbabilities(diffuse_probabilities)
    mgdm.setSteps(num_steps)
    mgdm.setTopology(topology)

    for idx, con1 in enumerate(con1_files):
        print("Input files and filetypes:")
        print(con1_type + ":\t" + con1.split(os.path.sep)[-1])

        # load volume
        img = load_volume(con1)
        data = img.get_data()
        affine = img.get_affine()
        header = img.get_header()
        resolution = [x.item() for x in header.get_zooms()]
        dimensions = data.shape

        # convert orientation information to mgdm slice and orientation info
        sliceorder, LR, AP, IS = _get_mgdm_orientation(affine, mgdm)
        mgdm.setOrientations(sliceorder, LR, AP, IS)

        # use the first image to set the dimensions and resolutions
        mgdm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
        mgdm.setResolutions(resolution[0], resolution[1], resolution[2])

        # input contrast 1 to mgdm
        mgdm.setContrastImage1(cbstools.JArray('float')((data.flatten('F')).astype(float)))
        mgdm.setContrastType1(con1_type)

        # TODO this seems like it could be simplified and then allow for
        # a variable number of contrast inputs
        if con2_files is not None:
            print(con2_type + ":\t" + con2_files[idx].split(os.path.sep)[-1])
            data = load_volume(con2_files[idx]).get_data()
            mgdm.setContrastImage2(cbstools.JArray('float')((data.flatten('F')).astype(float)))
            mgdm.setContrastType2(con2_type)

            if con3_files is not None:
                print(con3_type + ":\t" + con3_files[idx].split(os.path.sep)[-1])
                data = load_volume(con3_files[idx]).get_data()
                mgdm.setContrastImage3(cbstools.JArray('float')((data.flatten('F')).astype(float)))
                mgdm.setContrastType3(con3_type)

                if con4_files is not None:
                    print(con4_type + ":\t" + con4_files[idx].split(os.path.sep)[-1])
                    data = load_volume(con4_files[idx]).get_data()
                    mgdm.setContrastImage4(cbstools.JArray('float')((data.flatten('F')).astype(float)))
                    mgdm.setContrastType4(con4_type)

        # execute MGDM
        try:
            print("Executing MGDM on your inputs")
            mgdm.execute()

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

            out_files_seg.append(seg_nii)
            out_files_lbl.append(lbl_nii)
            out_files_ids.append(ids_nii)

            #  TODO: fix saving conventions
            #  save files
            # # set output dir if not given, create output dir if not exist
            # if output_dir is None:
            #     output_dir = os.path.dirname(con1_files[0])
            # create_dir(output_dir)
            if save_data:
                if base_name is None:
                    base_name = os.getcwd() + '/'
                    print "saving to %s" % base_name

                if file_suffix is not None:
                    seg_file = os.path.join(base_name +
                                            '_seg' + file_suffix + '.nii.gz')
                    lbl_file = os.path.join(base_name +
                                            '_lbl' + file_suffix + '.nii.gz')
                    ids_file = os.path.join(base_name +
                                            '_ids' + file_suffix + '.nii.gz')
                else:
                    seg_file = os.path.join(base_name + '_seg.nii.gz')
                    lbl_file = os.path.join(base_name + '_lbl.nii.gz')
                    ids_file = os.path.join(base_name + '_ids.nii.gz')

                save_volume(seg_file, seg_nii)
                save_volume(lbl_file, lbl_nii)
                save_volume(ids_file, ids_nii)

        except:
            # TODO: Catch specific errors and print informative messages
            print("Sorry, MGDM failed")
            return

    return out_files_seg, out_files_lbl, out_files_ids
