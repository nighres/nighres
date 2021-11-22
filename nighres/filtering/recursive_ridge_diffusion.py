import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, \
                    _check_available_memory


def recursive_ridge_diffusion(input_image, ridge_intensities, ridge_filter,
                              surface_levelset=None, orientation='undefined',
                              loc_prior=None,
                              min_scale=0, max_scale=3,
                              diffusion_factor=1.0,
                              similarity_scale=0.1,
                              max_iter=100, max_diff=1e-3,
                              threshold=0.5, 
                              save_data=False, overwrite=False, output_dir=None,
                              file_name=None):

    """ Recursive Ridge Diffusion

    Extracts planar of tubular structures across multiple scales, with an
    optional directional bias.


    Parameters
    ----------
    input_image: niimg
        Input image
    ridge_intensities: {'bright','dark','both'}
        Which intensities to consider for the filtering
    ridge_filter: {'2D','1D','0D'}
        Whether to filter for 2D ridges, 1D vessels, or 0D holes
    surface_levelset: niimg, optional
        Level set surface to restrict the orientation of the detected features
    orientation: {'undefined','parallel','orthogonal'}
        The orientation of features to keep with regard to the surface or its normal
    loc_prior: niimg, optional
        Location prior image to restrict the search for features
    min_scale: int
        Minimum scale (in voxels) to look for features (default is 0)
    max_scale: int
        Maximum scale (in voxels) to look for features (default is 3)
    diffusion_factor: float
        Scaling factor for the diffusion weighting in [0,1] (default is 1.0)
    similarity_scale: float
        Scaling of the similarity function as a factor of intensity range
    max_iter: int
        Maximum number of diffusion iterations
    max_diff: int
        Maximum difference to stop the diffusion
    threshold: float
        Detection threshold for the structures to keep (default is 0.5)
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
        (suffix of output files in brackets)

        * filter (niimg): raw filter response (_rrd-filter)
        * propagation (niimg): propagated probabilistic response after diffusion (_rrd-propag)
        * scale (niimg): scale of the detection filter  (_rrd-scale)
        * ridge_dir (niimg): estimated local ridge direction (_rrd-dir)
        * ridge_pv (niimg): ridge partial volume map, taking size into account (_rrd-pv)
        * ridge_size (niimg): estimated size of each detected component (rrd-size)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Extension of the recursive ridge
    filter in [1]_.

    References
    ----------
    .. [1] Bazin et al (2016), Vessel segmentation from quantitative
           susceptibility maps for local oxygenation venography, Proc ISBI.

    """

    print('\n Recursive Ridge Diffusion')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_image)

        filter_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd-filter'))

        propagation_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=input_image,
                                   suffix='rrd-propag'))

        scale_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=input_image,
                                   suffix='rrd-scale'))

        ridge_direction_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd-dir'))

        ridge_pv_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd-pv'))

        ridge_size_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='rrd-size'))

        if overwrite is False \
            and os.path.isfile(filter_file) \
            and os.path.isfile(propagation_file) \
            and os.path.isfile(scale_file) \
            and os.path.isfile(ridge_direction_file) \
            and os.path.isfile(ridge_pv_file) \
            and os.path.isfile(ridge_size_file) :

            print("skip computation (use existing results)")
            output = {'filter': filter_file,
                      'propagation': propagation_file,
                      'scale': scale_file,
                      'ridge_dir': ridge_direction_file,
                      'ridge_pv': ridge_pv_file,
                      'ridge_size': ridge_size_file}
            return output


    # load input image and use it to set dimensions and resolution
    img = load_volume(input_image)
    data = img.get_data()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    if (len(dimensions)<3): dimensions = (dimensions[0], dimensions[1], 1)
    if (len(resolution)<3): resolution = [resolution[0], resolution[1], 1.0]

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create extraction instance
    if dimensions[2]==1: rrd = nighresjava.FilterRecursiveRidgeDiffusion2D()
    else: rrd = nighresjava.FilterRecursiveRidgeDiffusion()

    # set parameters
    rrd.setRidgeIntensities(ridge_intensities)
    rrd.setRidgeFilter(ridge_filter)
    rrd.setOrientationToSurface(orientation)
    rrd.setMinimumScale(min_scale)
    rrd.setMaximumScale(max_scale)
    rrd.setDiffusionFactor(diffusion_factor)
    rrd.setSimilarityScale(similarity_scale)
    rrd.setPropagationModel("none")
    if max_iter>0: rrd.setPropagationModel("diffusion")
    rrd.setMaxIterations(max_iter)
    rrd.setMaxDifference(max_diff)
    rrd.setDetectionThreshold(threshold)
    #rrd.setUseRatio(use_ratio)

    rrd.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    rrd.setResolutions(resolution[0], resolution[1], resolution[2])

    # input input_image
    rrd.setInputImage(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))

    # input surface_levelset : dirty fix for the case where surface image not input
    try:
        data = load_volume(surface_levelset).get_data()
        rrd.setSurfaceLevelSet(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    except:
        print("no surface image")

    # input location prior image : loc_prior is optional
    try:
        data = load_volume(loc_prior).get_data()
        rrd.setLocationPrior(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    except:
        print("no location prior image")


    # execute Extraction
    try:
        rrd.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    filter_data = np.reshape(np.array(rrd.getFilterResponseImage(),
                                   dtype=np.float32), dimensions, 'F')

    propagation_data = np.reshape(np.array(rrd.getPropagatedResponseImage(),
                                    dtype=np.float32), dimensions, 'F')

    scale_data = np.reshape(np.array(rrd.getDetectionScaleImage(),
                                   dtype=np.int32), dimensions, 'F')

    if dimensions[2]==1:
        ridge_direction_data = np.reshape(np.array(rrd.getRidgeDirectionImage(),
                                    dtype=np.float32),
                                    (dimensions[0],dimensions[1],2),
                                    'F')
    else:
        ridge_direction_data = np.reshape(np.array(rrd.getRidgeDirectionImage(),
                                    dtype=np.float32),
                                    (dimensions[0],dimensions[1],dimensions[2],3),
                                    'F')

    ridge_pv_data = np.reshape(np.array(rrd.getRidgePartialVolumeImage(),
                                   dtype=np.float32), dimensions, 'F')

    ridge_size_data = np.reshape(np.array(rrd.getRidgeSizeImage(),
                                    dtype=np.float32), dimensions, 'F')


    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(filter_data)
    filter_img = nb.Nifti1Image(filter_data, affine, header)

    header['cal_max'] = np.nanmax(propagation_data)
    propag_img = nb.Nifti1Image(propagation_data, affine, header)

    header['cal_max'] = np.nanmax(scale_data)
    scale_img = nb.Nifti1Image(scale_data, affine, header)

    header['cal_max'] = np.nanmax(ridge_direction_data)
    ridge_dir_img = nb.Nifti1Image(ridge_direction_data, affine, header)

    header['cal_max'] = np.nanmax(ridge_pv_data)
    ridge_pv_img = nb.Nifti1Image(ridge_pv_data, affine, header)

    header['cal_max'] = np.nanmax(ridge_size_data)
    ridge_size_img = nb.Nifti1Image(ridge_size_data, affine, header)

    if save_data:
        save_volume(filter_file, filter_img)
        save_volume(propagation_file, propag_img)
        save_volume(scale_file, scale_img)
        save_volume(ridge_direction_file, ridge_dir_img)
        save_volume(ridge_pv_file, ridge_pv_img)
        save_volume(ridge_size_file, ridge_size_img)

        return {'filter': filter_file, 'propagation': propagation_file, 'scale': scale_file,
                'ridge_dir': ridge_direction_file, 'ridge_pv': ridge_pv_file,
                'ridge_size': ridge_size_file}
    else:
        return {'filter': filter_img, 'propagation': propag_img, 'scale': scale_img,
                'ridge_dir': ridge_dir_img, 'ridge_pv': ridge_pv_img,
                'ridge_size': ridge_size_img}
