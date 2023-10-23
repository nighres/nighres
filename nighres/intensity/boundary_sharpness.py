import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def boundary_sharpness(image, mask=None, scaling=16.0, noise_level=0.002, iterations=-1,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Boundary Sharpness

    Estimates the sharpness of boundaries inside an image based on super-voxel parcellation [1]_
    and fitting a sigmoid to each corresponding boundary [2]_

    Parameters
    ----------
    image: niimg
        Input image
    mask: niimg, optional
        Data mask to specify acceptable seeding regions
    scaling: float, optional
        Scaling factor for the new super-voxel grid (default is 4)
    noise_level: float, optional
        Weighting parameter to balance image intensity and spatial variability
    iterations: int, optional
        Maximum number of iterations in the boundary estimate (default is 10)
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

        * parcel (niimg): The super-voxel parcellation of the original image, with average 
            intensity as label
        * boundaries (niimg): The probability score of the super-voxel boundaries
        * cnr (niimg): The contrast-to-noise ratio of the super-voxel boundaries
        * sharpness (niimg): The slope of the sigmoid representing each boundary

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    
    References
    ----------
    .. [1] R. Achanta and S. Suesstrunk, 
        Superpixels and Polygons using Simple Non-Iterative Clustering,
        Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2017.

    .. [2] P.L. Bazin, H.E. Nijsse, W.van der Zwaag, D. Gallichan,
        A. Alkemade, F.M. Vos, B.U. Forstmann, M.W.A. Caan
        Sharpness in motion corrected quantitative imaging at 7T,
        NeuroImage, 2020.

    """

    print('\nBoundary Sharpness')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        parcel_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='bshp-parcel'))

        boundaries_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='bshp-boundaries'))

        cnr_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='bshp-cnr'))

        sharpness_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='bshp-sharpness'))

        if overwrite is False \
            and os.path.isfile(parcel_file) \
            and os.path.isfile(boundaries_file) \
            and os.path.isfile(cnr_file) \
            and os.path.isfile(sharpness_file) :
                print("skip computation (use existing results)")
                output = {'parcel': parcel_file, 'boundaries': boundaries_file,
                          'cnr': cnr_file, 'sharpness': sharpness_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    bsharp = nighresjava.BoundarySharpness()

    # set parameters
    
    # load image and use it to set dimensions and resolution
    img = load_volume(image)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    if len(dimensions)>2:
        bsharp.setDimensions(dimensions[0], dimensions[1], dimensions[2])
        bsharp.setResolutions(resolution[0], resolution[1], resolution[2])
    else:
        bsharp.setDimensions(dimensions[0], dimensions[1])
        bsharp.setResolutions(resolution[0], resolution[1])
        
    bsharp.setInputImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    
    if mask is not None:
        bsharp.setMaskImage(nighresjava.JArray('int')(
                (load_volume(mask).get_fdata().flatten('F')).astype(int).tolist()))
    
    # set algorithm parameters
    bsharp.setScalingFactor(scaling)
    bsharp.setNoiseLevel(noise_level)
    bsharp.setIterations(iterations)
    
    # execute the algorithm
    try:
        bsharp.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    parcel_data = np.reshape(np.array(bsharp.getParcelImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(parcel_data)
    header['cal_max'] = np.nanmax(parcel_data)
    parcel = nb.Nifti1Image(parcel_data, affine, header)

    boundaries_data = np.reshape(np.array(bsharp.getBoundariesImage(),
                                    dtype=np.float32), dimensions, 'F')

    header['cal_min'] = np.nanmin(boundaries_data)
    header['cal_max'] = np.nanmax(boundaries_data)
    boundaries = nb.Nifti1Image(boundaries_data, affine, header)

    cnr_data = np.reshape(np.array(bsharp.getCNRImage(),
                                    dtype=np.float32), dimensions, 'F')

    header['cal_min'] = np.nanmin(cnr_data)
    header['cal_max'] = np.nanmax(cnr_data)
    cnr = nb.Nifti1Image(cnr_data, affine, header)

    sharpness_data = np.reshape(np.array(bsharp.getSharpnessImage(),
                                    dtype=np.float32), dimensions, 'F')

    header['cal_min'] = np.nanmin(sharpness_data)
    header['cal_max'] = np.nanmax(sharpness_data)
    sharpness = nb.Nifti1Image(sharpness_data, affine, header)

    if save_data:
        save_volume(parcel_file, parcel)
        save_volume(boundaries_file, boundaries)
        save_volume(cnr_file, cnr)
        save_volume(sharpness_file, sharpness)
        return {'parcel': parcel_file, 'boundaries': boundaries_file,
                'cnr': cnr_file, 'sharpness': sharpness_file}
        return {'parcel': parcel, 'boundaries': boundaries, 'cnr': cnr, 'shrapness': sharpness}
