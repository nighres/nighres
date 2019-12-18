import os
import sys
import numpy
import nibabel
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, _check_available_memory


def laminar_regional_approximation(profile_surface_image, intensity_image, roi_image,
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
    roi_image: niimg
        Label image of the region to sample
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

        * weights (niimg): weight image, representing the weighting of profiles in the estimation (_lra-weight)
        * degree (niimg): degree image, representing the degree of profiles in the estimation (_lra-deg)
        * residuals (niimg): residuals image, representing the residual error in the estimation (_lra-res)
        * median ([float]): the estimated median profile (_lra-med)
        * perc25 ([float]): the estimated 25 percentile profile (_lra-p25)
        * perc75 ([float]): the estimated 75 percentile profile (_lra-p75)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin
    '''

    print('\nProfile regional approximation')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, intensity_image)

        weight_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                      rootfile=intensity_image,
                                      suffix='lra-weight'))
        deg_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                      rootfile=intensity_image,
                                      suffix='lra-deg'))
        res_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                      rootfile=intensity_image,
                                      suffix='lra-res'))
        perc25_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                      rootfile=intensity_image,
                                      suffix='lra-p25',ext='txt'))
        median_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                      rootfile=intensity_image,
                                      suffix='lpa-med',ext='txt'))
        perc75_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                      rootfile=intensity_image,
                                      suffix='lpa-p75',ext='txt'))
        if overwrite is False \
            and os.path.isfile(weight_file) and os.path.isfile(sample_file) \
            and os.path.isfile(median_file) and os.path.isfile(iqr_file) :

            print("skip computation (use existing results)")
            output = {'weights': weight_file,
                      'best': numpy.loadtxt(sample_file),
                      'median': numpy.loadtxt(median_file),
                      'iqr': numpy.loadtxt(iqr_file)}
            return output

    # start VM if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initate class
    sampler = nighresjava.LaminarProfileAveraging()

    # load the data
    surface_img = load_volume(profile_surface_image)
    surface_data = surface_img.get_data()
    hdr = surface_img.header
    aff = surface_img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = surface_data.shape

    intensity_data = load_volume(intensity_image).get_data()

    roi_data = load_volume(roi_image).get_data()

    # pass inputs
    sampler.setIntensityImage(nighresjava.JArray('float')(
                                  (intensity_data.flatten('F')).astype(float)))
    sampler.setProfileSurfaceImage(nighresjava.JArray('float')(
                                   (surface_data.flatten('F')).astype(float)))
    sampler.setRoiMask(nighresjava.JArray('int')(
                                   (roi_data.flatten('F')).astype(int).tolist()))
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
    weight_data = numpy.reshape(numpy.array(
                                sampler.getProfileWeights(),
                                dtype=numpy.float32), (dimensions[0],dimensions[1],dimensions[2]), 'F')

    sample = numpy.array(sampler.getSampleProfile(), dtype=numpy.float32)
    median = numpy.array(sampler.getMedianProfile(), dtype=numpy.float32)
    iqr = numpy.array(sampler.getIqrProfile(), dtype=numpy.float32)

    hdr['cal_max'] = numpy.nanmax(weight_data)
    weights = nibabel.Nifti1Image(weight_data, aff, hdr)

    if save_data:
        save_volume(weight_file, weights)
        numpy.savetxt(sample_file, sample)
        numpy.savetxt(median_file, median)
        numpy.savetxt(iqr_file, iqr)
        return {'weights': weight_file}
    else:
        return {'weights': weights}
