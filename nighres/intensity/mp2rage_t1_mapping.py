import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir


def flash_t2s_fitting(first_inversion, second_inversion, 
                      inversion_times, flip_angles, inversion_TR,
                      excitation_TR, N_excitations, efficiency=0.96,
                      correct_B1=False, B1_map=None,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ MP2RAGE T1 mapping

    Estimate T1/R1 by a look-up table method adapted from [1]_

    Parameters
    ----------
    first_inversion: [niimg]
        List of {magnitude, phase} images for the first inversion
    second_inversion: [niimg]
        List of {magnitude, phase} images for the second inversion
    inversion_times: [float]
        List of {first, second} inversion times
    flip_angles: [float]
        List of {first, second} flip angles
    inversion_TR: float
        Inversion repetition time
    excitation_TR: [float]
        List of {first,second} repetition times
    N_excitations: int
        Number of excitations
    efficiency: float
        Inversion efficiency (default is 0.96)
    correct_B1: bool
        Whether to correct for B1 inhomogeneities (default is False)
    B1_map: niimg
        Computed B1 map
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

        * t1 (niimg): Map of estimated T1 times (_qt1map-t1)
        * r1 (niimg): Map of estimated R1 relaxation rate (_qt1map-r1)
        * uni (niimg): Estimated PD weighted image at TE=0 (_qt1map-uni)
        
    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    
    References
    ----------
    .. [1] Marques, Kober, Krueger, van der Zwaag, Van de Moortele, Gruetter (2010)
        MP2RAGE, a self bias-field corrected sequence for improved segmentation 
        and T1-mapping at high field. doi: 10.1016/j.neuroimage.2009.10.002.
    """

    print('\nT1 Mapping')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image_list[0])

        t1_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=image_list[0],
                                   suffix='qt1map-t1'))

        r1_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=image_list[0],
                                   suffix='qt1map-r1'))

        uni_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=image_list[0],
                                   suffix='qt1map-uni'))

        if overwrite is False \
            and os.path.isfile(t1_file) \
            and os.path.isfile(r1_file) \
            and os.path.isfile(uni_file) :
                output = {'t1': load_volume(t1_file),
                          'r1': load_volume(r1_file), 
                          'uni': load_volume(uni_file)}
                return output

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='12000m', maxheap='12000m')
    except ValueError:
        pass
    # create algorithm instance
    qt1map = cbstools.IntensityMp2rageT1Fitting()

    # set algorithm parameters
    qt1map.setFirstInversionTime(inversion_times[0])
    qt1map.setSecondInversionTime(inversion_times[1])
    qt1map.setFirstFlipAngle(flip_angles[0])
    qt1map.setSecondFlipAngle(flip_angles[1])
    qt1map.setInversionRepetitionTime(inversion_TR)
    qt1map.setFirstExcitationRepetitionTime(excitation_TR[0])
    qt1map.setSecondExcitationRepetitionTime(excitation_TR[1])
    qt1map.setNumberExcitations(N_excitations)
    qt1map.setInversionEfficiency(efficiency)
    qt1map.setCorrectB1inhomogeneities(correct_B1)
     
    # load first image and use it to set dimensions and resolution
    img = load_volume(first_inversion[0])
    data = img.get_data()
    #data = data[0:10,0:10,0:10]
    affine = img.get_affine()
    header = img.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    qt1map.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    qt1map.setResolutions(resolution[0], resolution[1], resolution[2])

    # input images
    qt1map.setFirstInversionMagnitude(cbstools.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    data = load_volume(first_inversion[1]).get_data()
    qt1map.setFirstInversionPhase(cbstools.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    data = load_volume(second_inversion[0]).get_data()
    qt1map.setFirstInversionMagnitude(cbstools.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    data = load_volume(second_inversion[1]).get_data()
    qt1map.setFirstInversionPhase(cbstools.JArray('float')(
                                    (data.flatten('F')).astype(float)))
 
    if (correct_B1):
        data = load_volume(B1_map).get_data()
        qt1map.setB1mapImage(cbstools.JArray('float')(
                                    (data.flatten('F')).astype(float)))
 
    # execute the algorithm
    try:
        qt1map.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    t2s_data = np.reshape(np.array(qt2fit.getT2sImage(),
                                    dtype=np.float32), dimensions, 'F')

    r2s_data = np.reshape(np.array(qt2fit.getR2sImage(),
                                    dtype=np.float32), dimensions, 'F')

    s0_data = np.reshape(np.array(qt2fit.getS0Image(),
                                    dtype=np.float32), dimensions, 'F')

    err_data = np.reshape(np.array(qt2fit.getResidualImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(t2s_data)
    header['cal_max'] = np.nanmax(t2s_data)
    t2s = nb.Nifti1Image(t2s_data, affine, header)

    header['cal_min'] = np.nanmin(r2s_data)
    header['cal_max'] = np.nanmax(r2s_data)
    r2s = nb.Nifti1Image(r2s_data, affine, header)

    header['cal_min'] = np.nanmin(s0_data)
    header['cal_max'] = np.nanmax(s0_data)
    s0 = nb.Nifti1Image(s0_data, affine, header)

    header['cal_min'] = np.nanmin(err_data)
    header['cal_max'] = np.nanmax(err_data)
    err = nb.Nifti1Image(err_data, affine, header)

    if save_data:
        save_volume(t2s_file, t2s)
        save_volume(r2s_file, r2s)
        save_volume(s0_file, s0)
        save_volume(err_file, err)

    return {'t2s': t2s, 'r2s': r2s, 's0': s0, 'residuals': err}