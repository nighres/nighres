import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_atlas_file


def _get_mgdm_orientation(affine, mgdm):
    '''
    Transforms nibabel affine information into
    orientation and slice order that MGDM understands
    '''
    orientation = nb.aff2axcodes(affine)
    # set mgdm slice order
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
                      n_steps=5, max_iterations=800, topology='wcs',
                      atlas_file=None, topology_lut_dir=None,
                      adjust_intensity_priors=False,
                      compute_posterior=False,
                      diffuse_probabilities=False,
                      save_data=False, output_dir=None,
                      file_name=None):
    """ MGDM segmentation

    Estimates brain structures from an atlas for MRI data using
    a Multiple Object Geometric Deformable Model (MGDM)

    Parameters
    ----------
    contrast_image1: niimg
        First input image to perform segmentation on
    contrast_type1: str
        Contrast type of first input image, must be listed as a prior in used
        atlas(specified in atlas_file)
    contrast_image2: niimg, optional
        Additional input image to inform segmentation, must be in the same
        space as constrast_image1, requires contrast_type2
    contrast_type2: str, optional
        Contrast type of second input image, must be listed as a prior in used
        atlas (specified in atlas_file)
    contrast_image3: niimg, optional
        Additional input image to inform segmentation, must be in the same
        space as constrast_image1, requires contrast_type3
    contrast_type3: str, optional
        Contrast type of third input image, must be listed as a prior in used
        atlas (specified in atlas_file)
    contrast_image4: niimg, optional
        Additional input image to inform segmentation, must be in the same
        space as constrast_image1, requires contrast_type4
    contrast_type4: str, optional
        Contrast type of fourth input image, must be listed as a prior in used
        atlas (specified in atlas_file)
    n_steps: int, optional
        Number of steps for MGDM (default is 5, set to 0 for quick testing of
        registration of priors, which does not perform true segmentation)
    max_iterations: int, optional
        Maximum number of iterations per step for MGDM (default is 800, set
        to 1 for quick testing of registration of priors, which does not
        perform true segmentation)
    topology: {'wcs', 'no'}, optional
        Topology setting, choose 'wcs' (well-composed surfaces) for strongest
        topology constraint, 'no' for no topology constraint (default is 'wcs')
    atlas_file: str, optional
        Path to plain text atlas file (default is stored in DEFAULT_ATLAS)
        or atlas name to be searched in ATLAS_DIR
    topology_lut_dir: str, optional
        Path to directory in which topology files are stored (default is stored
        in TOPOLOGY_LUT_DIR)
    adjust_intensity_priors: bool
        Adjust intensity priors based on dataset (default is False)
    compute_posterior: bool
        Compute posterior probabilities for segmented structures
        (default is False)
    diffuse_probabilities: bool
        Regularize probability distribution with a non-linear diffusion scheme
        (default is False)
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

        * segmentation (niimg): Hard brain segmentation with topological
          constraints (if chosen) (_mgdm_seg)
        * labels (niimg): Maximum tissue probability labels (_mgdm_lbls)
        * memberships (niimg): Maximum tissue probability values, 4D image
          where the first dimension shows each voxel's highest probability to
          belong to a specific tissue, the second dimension shows the second
          highest probability to belong to another tissue etc. (_mgdm_mems)
        * distance (niimg): Minimum distance to a segmentation boundary
          (_mgdm_dist)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Algorithm details can be
    found in [1]_ and [2]_

    References
    ----------
    .. [1] Bogovic, Prince and Bazin (2013). A multiple object geometric
       deformable model for image segmentation.
       doi:10.1016/j.cviu.2012.10.006.A
    .. [2] Fan, Bazin and Prince (2008). A multi-compartment segmentation
       framework with homeomorphic level sets. DOI: 10.1109/CVPR.2008.4587475
    """

    print('\nMGDM Segmentation')

    # check atlas_file and set default if not given
    atlas_file = _check_atlas_file(atlas_file)

    # check topology_lut_dir and set default if not given
    topology_lut_dir = _check_topology_lut_dir(topology_lut_dir)

    # find available intensity priors in selected MGDM atlas
    mgdm_intensity_priors = _get_mgdm_intensity_priors(atlas_file)

    # sanity check contrast types
    contrasts = [contrast_image1, contrast_image2,
                 contrast_image3, contrast_image4]
    ctypes = [contrast_type1, contrast_type2, contrast_type3, contrast_type4]
    for idx, ctype in enumerate(ctypes):
        if ctype is None and contrasts[idx] is not None:
            raise ValueError(("If specifying contrast_image{0}, please also "
                              "specify contrast_type{0}".format(idx+1, idx+1)))

        elif ctype is not None and ctype not in mgdm_intensity_priors:
            raise ValueError(("{0} is not a valid contrast type for  "
                              "contrast_type{1} please choose from the "
                              "following contrasts provided by the chosen "
                              "atlas: ").format(ctype, idx+1),
                             ", ".join(mgdm_intensity_priors))

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, contrast_image1)

        seg_file = _fname_4saving(file_name=file_name,
                                  rootfile=contrast_image1,
                                  suffix='mgdm_seg', )

        lbl_file = _fname_4saving(file_name=file_name,
                                  rootfile=contrast_image1,
                                  suffix='mgdm_lbls')

        mems_file = _fname_4saving(file_name=file_name,
                                   rootfile=contrast_image1,
                                   suffix='mgdm_mems')

        dist_file = _fname_4saving(file_name=file_name,
                                   rootfile=contrast_image1,
                                   suffix='mgdm_dist')

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
    mgdm.setOutputImages('label_memberships')
    mgdm.setAdjustIntensityPriors(adjust_intensity_priors)
    mgdm.setComputePosterior(compute_posterior)
    mgdm.setDiffuseProbabilities(diffuse_probabilities)
    mgdm.setSteps(n_steps)
    mgdm.setMaxIterations(max_iterations)
    mgdm.setTopology(topology)
    mgdm.setNormalizeQuantitativeMaps(True)
    # set to False for "quantitative" brain prior atlases
    # (version quant-3.0.5 and above)

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
    mgdm.setContrastImage1(cbstools.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    mgdm.setContrastType1(contrast_type1)

    # if further contrast are specified, input them
    if contrast_image2 is not None:
        data = load_volume(contrast_image2).get_data()
        mgdm.setContrastImage2(cbstools.JArray('float')(
                                            (data.flatten('F')).astype(float)))
        mgdm.setContrastType2(contrast_type2)

        if contrast_image3 is not None:
            data = load_volume(contrast_image3).get_data()
            mgdm.setContrastImage3(cbstools.JArray('float')(
                                            (data.flatten('F')).astype(float)))
            mgdm.setContrastType3(contrast_type3)

            if contrast_image4 is not None:
                data = load_volume(contrast_image4).get_data()
                mgdm.setContrastImage4(cbstools.JArray('float')(
                                            (data.flatten('F')).astype(float)))
                mgdm.setContrastType4(contrast_type4)

    # execute MGDM
    try:
        mgdm.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

    # reshape output to what nibabel likes
    seg_data = np.reshape(np.array(mgdm.getSegmentedBrainImage(),
                                   dtype=np.int32), dimensions, 'F')

    dist_data = np.reshape(np.array(mgdm.getLevelsetBoundaryImage(),
                                    dtype=np.float32), dimensions, 'F')

    # membership and labels output has a 4th dimension, set to 6
    dimensions4d = [dimensions[0], dimensions[1], dimensions[2], 6]
    lbl_data = np.reshape(np.array(mgdm.getPosteriorMaximumLabels4D(),
                                   dtype=np.int32), dimensions4d, 'F')
    mems_data = np.reshape(np.array(mgdm.getPosteriorMaximumMemberships4D(),
                                    dtype=np.float32), dimensions4d, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(seg_data)
    seg = nb.Nifti1Image(seg_data, affine, header)

    header['cal_max'] = np.nanmax(dist_data)
    dist = nb.Nifti1Image(dist_data, affine, header)

    header['cal_max'] = np.nanmax(lbl_data)
    lbls = nb.Nifti1Image(lbl_data, affine, header)

    header['cal_max'] = np.nanmax(mems_data)
    mems = nb.Nifti1Image(mems_data, affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, seg_file), seg)
        save_volume(os.path.join(output_dir, dist_file), dist)
        save_volume(os.path.join(output_dir, lbl_file), lbls)
        save_volume(os.path.join(output_dir, mems_file), mems)

    return {'segmentation': seg, 'labels': lbls,
            'memberships': mems, 'distance': dist}
