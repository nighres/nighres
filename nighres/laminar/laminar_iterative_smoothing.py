import os
import numpy as np
import nibabel as nb
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving


def laminar_iterative_smoothing(profile_surface_image, intensity_image, fwhm_mm,
                     roi_mask_image=None, 
                     save_data=False, overwrite=False, output_dir=None,
                     file_name=None):

    '''Smoothing data on multiple intracortical layers

    Parameters
    -----------
    data_image: niimg
        Image from which data should be sampled
    profile_surface_image: niimg
        4D image containing levelset representations of different intracortical
        surfaces on which data should be sampled
    fwhm_mm: float
        Full width half maximum distance to use in smoothing (in mm)
    roi_mask_image: niimg, optional
        Mask image defining a region of interest to restrict the smoothing
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
    -----------
    niimg
        smoothed intensity image (output file suffix _lis_smooth)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin
    
    Important: this method assumes isotropic voxels
    '''

    print('\nLaminar iterative smoothing')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, intensity_image)

        smoothed_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                      rootfile=intensity_image,
                                      suffix='lis_smooth'))

    # start VM if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass

    # initate class
    smoother = cbstools.LaminarIterativeSmoothing()

    # load the data
    surface_img = load_volume(profile_surface_image)
    surface_data = surface_img.get_data()
    layers = surface_data.shape[3]-1

    intensity_img = load_volume(intensity_image)
    intensity_data = intensity_img.get_data()
    hdr = intensity_img.get_header()
    aff = intensity_img.get_affine()
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = intensity_data.shape
    
    if (roi_mask_image!=None) :
        roi_mask_data = load_volume(data_image).get_data()
    else :
        roi_mask_data = None

    # pass inputs
    smoother.setIntensityImage(cbstools.JArray('float')(
                                  (intensity_data.flatten('F')).astype(float)))
    smoother.setProfileSurfaceImage(cbstools.JArray('float')(
                                   (surface_data.flatten('F')).astype(float)))
    smoother.setResolutions(resolution[0], resolution[1], resolution[2])
    smoother.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    smoother.setLayers(layers)
    if (len(dimensions)>3) :
        smoother.set4thDimension(dimensions[3])
    else :
        smoother.set4thDimension(1)

    if (roi_mask_data!=None): 
        smoother.setROIMask(cbstools.JArray('int')(
                                  (roi_mask_data.flatten('F')).astype(int).tolist()))
    smoother.setFWHMmm(float(fwhm_mm))

    # execute class
    try:
        smoother.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # collecting outputs
    smoothed_data = np.reshape(np.array(
                                smoother.getSmoothedIntensityImage(),
                                dtype=np.float32), dimensions, 'F')

    hdr['cal_max'] = np.nanmax(smoothed_data)
    smoothed = nb.Nifti1Image(smoothed_data, aff, hdr)

    if save_data:
        save_volume(os.path.join(output_dir, smoothed_file), smoothed)

    return smoothed
