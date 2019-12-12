import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def mgdm_cells(contrast_image1, contrast_type1,
                      contrast_image2=None, contrast_type2=None,
                      contrast_image3=None, contrast_type3=None,
                      stack_dimension='2D',
                      force_weight=0.6, curvature_weight=0.3,
                      cell_threshold=0.1,
                      max_iterations=200, min_change=0.0001,
                      topology='wcs', topology_lut_dir=None,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ MGDM cell segmentation

    Estimates cell structures using
    a Multiple Object Geometric Deformable Model (MGDM)

    Parameters
    ----------
    contrast_image1: niimg
        First input image to perform segmentation on
    contrast_type1: {"centroid-proba", "local-maxima","foreground-proba",
        "image-intensities"}
        Contrast type of first input image
    contrast_image2: niimg, optional
        Additional input image to inform segmentation, must be in the same
        space as constrast_image1, requires contrast_type2
    contrast_type2: str, {"centroid-proba", "local-maxima","foreground-proba",
        "image-intensities"}, optional
        Contrast type of second input image
    contrast_image3: niimg, optional
        Additional input image to inform segmentation, must be in the same
        space as constrast_image1, requires contrast_type3
    contrast_type3: {"centroid-proba", "local-maxima","foreground-proba",
        "image-intensities"}, optional
        Contrast type of third input image
    stack_dimension: {'2D','3D'}, optional
        Dimension of the data for processing, either a stack of independent
        2D slices or a fully 3D stack
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
    cell_threshold: float, optional
        Ratio of lower intensities from the local maximum to be included in
        a given cell (default is 0.1)
    topology: {'wcs', 'no'}, optional
        Topology setting, choose 'wcs' (well-composed surfaces) for strongest
        topology constraint, 'no' for no topology constraint (default is 'wcs')
    topology_lut_dir: str, optional
        Path to directory in which topology files are stored (default is stored
        in TOPOLOGY_LUT_DIR)
    save_data: bool, optional
        Save output data to file (default is False)
    overwrite: bool, optional
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
        (suffix of output files in brackets)

        * segmentation (niimg): Hard brain segmentation with topological
          constraints (if chosen) (_mgdmc-seg)
        * distance (niimg): Minimum distance to a segmentation boundary
          (_mgdmc-dist)

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

        seg_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=contrast_image1,
                                  suffix='mgdmc-seg', ))

        dist_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=contrast_image1,
                                   suffix='mgdmc-dist'))
        if overwrite is False \
            and os.path.isfile(seg_file) \
            and os.path.isfile(dist_file) :

            print("skip computation (use existing results)")
            output = {'segmentation': seg_file,
                      'distance': dist_file}
            return output


    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create mgdm instance
    mgdm = nighresjava.SegmentationCellMgdm()

    # set mgdm parameters
    mgdm.setDataStackDimension(stack_dimension)
    mgdm.setTopologyLUTdirectory(topology_lut_dir)
    mgdm.setMaxIterations(max_iterations)
    mgdm.setMinChange(min_change)
    mgdm.setDataWeight(force_weight)
    mgdm.setCurvatureWeight(curvature_weight)
    mgdm.setCellThreshold(cell_threshold)

    mgdm.setTopology(topology)

    # load contrast image 1 and use it to set dimensions and resolution
    img = load_volume(contrast_image1)
    data = img.get_data()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    mgdm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    mgdm.setResolutions(resolution[0], resolution[1], resolution[2])

    # input image 1
    mgdm.setContrastImage1(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    mgdm.setContrastType1(contrast_type1)

    # if further contrast are specified, input them
    if contrast_image2 is not None:
        data = load_volume(contrast_image2).get_data()
        mgdm.setContrastImage2(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
        mgdm.setContrastType2(contrast_type2)

        if contrast_image3 is not None:
            data = load_volume(contrast_image3).get_data()
            mgdm.setContrastImage3(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
            mgdm.setContrastType3(contrast_type3)

    # execute MGDM
    try:
        mgdm.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
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
        save_volume(seg_file, seg)
        save_volume(dist_file, dist)
        return {'segmentation': seg_file, 'distance': dist_file}
    else:
        return {'segmentation': seg, 'distance': dist}
