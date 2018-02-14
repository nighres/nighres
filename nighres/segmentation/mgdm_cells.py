import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir


def mgdm_cells(contrast_image1, contrast_type1,
                      contrast_image2=None, contrast_type2=None,
                      contrast_image3=None, contrast_type3=None,
                      force_weight=0.5, curvature_weight=0.2,
                      max_iterations=800, min_change=0.0001,
                      topology='wcs', topology_lut_dir=None,
                      save_data=False, output_dir=None,
                      file_name=None):
    """ MGDM segmentation

    Estimates cell structures using
    a Multiple Object Geometric Deformable Model (MGDM)

    Parameters
    ----------
    contrast_image1: niimg
        First input image to perform segmentation on
    contrast_type1: str
        Contrast type of first input image, must be in {"centroid-proba", 
        "local-maxima","foreground-proba","image-intensities"}
    contrast_image2: niimg, optional
        Additional input image to inform segmentation, must be in the same
        space as constrast_image1, requires contrast_type2
    contrast_type2: str, optional
        Contrast type of second input image, must be in {"centroid-proba", 
        "local-maxima","foreground-proba","image-intensities"}
    contrast_image3: niimg, optional
        Additional input image to inform segmentation, must be in the same
        space as constrast_image1, requires contrast_type3
    contrast_type3: str, optional
        Contrast type of third input image, must be in {"centroid-proba", 
        "local-maxima","foreground-proba","image-intensities"}
    max_iterations: int, optional
        Maximum number of iterations per step for MGDM (default is 800, set
        to 1 for quick testing of registration of priors, which does not
        perform true segmentation)
    min_change: float, optional
        Minimum amount of change in the segmentation for MGDM to stop 
        (default is 0.001)
    force_weight: float, optional
        Forces to drive MGDM to cell boundaries
        (default is 0.5)
    curvature_weight: float, optional
        Curvature regularization forces
        (default is 0.1)
    topology: {'wcs', 'no'}, optional
        Topology setting, choose 'wcs' (well-composed surfaces) for strongest
        topology constraint, 'no' for no topology constraint (default is 'wcs')
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

        * segmentation (niimg): Hard brain segmentation with topological
          constraints (if chosen) (_mgdmc_seg)
        * distance (niimg): Minimum distance to a segmentation boundary
          (_mgdmc_dist)

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

    print('\nMGDM Cell Segmentation')

    # check topology_lut_dir and set default if not given
    topology_lut_dir = _check_topology_lut_dir(topology_lut_dir)
  
    # sanity check contrast types
    contrasts = [contrast_image1, contrast_image2, contrast_image3]
    ctypes = [contrast_type1, contrast_type2, contrast_type3]
    for idx, ctype in enumerate(ctypes):
        if ctype is None and contrasts[idx] is not None:
            raise ValueError(("If specifying contrast_image{0}, please also "
                              "specify contrast_type{0}".format(idx+1, idx+1)))

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, contrast_image1)

        seg_file = _fname_4saving(file_name=file_name,
                                  rootfile=contrast_image1,
                                  suffix='mgdmc_seg', )

        dist_file = _fname_4saving(file_name=file_name,
                                   rootfile=contrast_image1,
                                   suffix='mgdmc_dist')

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass
    # create mgdm instance
    mgdm = cbstools.SegmentationCellMgdm()

    # set mgdm parameters
    mgdm.setTopologyLUTdirectory(topology_lut_dir)
    mgdm.setMaxIterations(max_iterations)
    mgdm.setMinChange(min_change)
    mgdm.setDataWeight(force_weight)
    mgdm.setCurvatureWeight(curvature_weight)
    
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
    seg_data = np.reshape(np.array(mgdm.getSegmentedImage(),
                                   dtype=np.int32), dimensions, 'F')

    dist_data = np.reshape(np.array(mgdm.getLevelsetBoundaryImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(seg_data)
    seg = nb.Nifti1Image(seg_data, affine, header)

    header['cal_max'] = np.nanmax(dist_data)
    dist = nb.Nifti1Image(dist_data, affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, seg_file), seg)
        save_volume(os.path.join(output_dir, dist_file), dist)

    return {'segmentation': seg, 'distance': dist}
