import os
import sys
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, _check_available_memory


def profile_sampling(profile_surface_image, intensity_image,
                     save_data=False, overwrite=False, output_dir=None,
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
    overwrite: bool
        Overwrite existing results (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)

    Returns
    -----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * result (niimg): 4D profile image , where the 4th dimension represents
          the profile for each voxel (_lps-data)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin and Juliane Dinse
    '''

    print('\nProfile sampling')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, intensity_image)

        profile_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                      rootfile=intensity_image,
                                      suffix='lps-data'))
        if overwrite is False \
            and os.path.isfile(profile_file) :

            print("skip computation (use existing results)")
            output = {'result': profile_file}
            return output

    # start VM if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initate class
    sampler = nighresjava.LaminarProfileSampling()

    # load the data
    surface_img = load_volume(profile_surface_image)
    surface_data = surface_img.get_data()
    hdr = surface_img.header
    aff = surface_img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = surface_data.shape

    intensity_data = load_volume(intensity_image).get_data()

    # pass inputs
    sampler.setIntensityImage(nighresjava.JArray('float')(
                                  (intensity_data.flatten('F')).astype(float)))
    sampler.setProfileSurfaceImage(nighresjava.JArray('float')(
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
        print(sys.exc_info()[0])
        raise
        return

    # collecting outputs
    profile_data = np.reshape(np.array(
                                sampler.getProfileMappedIntensityImage(),
                                dtype=np.float32), dimensions, 'F')

    hdr['cal_max'] = np.nanmax(profile_data)
    profiles = nb.Nifti1Image(profile_data, aff, hdr)

    if save_data:
        save_volume(profile_file, profiles)
        return {'result': profile_file}
    else:
        return {'result': profiles}
