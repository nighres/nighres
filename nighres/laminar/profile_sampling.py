import os
import numpy as np
import nibabel as nb
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving


def profile_sampling(profile_surface_image, intensity_image,
                     save_data=False, output_dir=None,
                     file_name=None):

    '''Sampling data on multiple intracortical layers

    Parameters
    -----------
    profile_surface_image: niimg
        4D image containing levelset representations of different intracortical
        surfaces on which data should be sampled
    intensity_image: niimg
        Image from which data should be sampled
    save_data: bool
        Save output data to file (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)

    Returns
    -----------
    niimg
        4D profile image , where the 4th dimension represents the
        profile for each voxel (output file suffix _profiles)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin and Juliane Dinse
    '''

    print('\nProfile sampling')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, intensity_image)

        profile_file = _fname_4saving(file_name=file_name,
                                      rootfile=intensity_image,
                                      suffix='profiles')

    # start VM if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass

    # initate class
    sampler = cbstools.LaminarProfileSampling()

    # load the data
    surface_img = load_volume(profile_surface_image)
    surface_data = surface_img.get_data()
    hdr = surface_img.get_header()
    aff = surface_img.get_affine()
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = surface_data.shape

    intensity_data = load_volume(intensity_image).get_data()

    # pass inputs
    sampler.setIntensityImage(cbstools.JArray('float')(
                                  (intensity_data.flatten('F')).astype(float)))
    sampler.setProfileSurfaceImage(cbstools.JArray('float')(
                                   (surface_data.flatten('F')).astype(float)))
    sampler.setResolutions(resolution[0], resolution[1], resolution[2])
    sampler.setDimensions(dimensions[0], dimensions[1],
                          dimensions[2], dimensions[3])

    # execute class
    try:
        sampler.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

    # collecting outputs
    profile_data = np.reshape(np.array(
                                sampler.getProfileMappedIntensityImage(),
                                dtype=np.float32), dimensions, 'F')

    hdr['cal_max'] = np.nanmax(profile_data)
    profiles = nb.Nifti1Image(profile_data, aff, hdr)

    if save_data:
        save_volume(os.path.join(output_dir, profile_file), profiles)

    return profiles
