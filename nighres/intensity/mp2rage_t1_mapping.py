import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def mp2rage_t1_mapping(first_inversion, second_inversion, 
                      inversion_times, flip_angles, inversion_TR,
                      excitation_TR, N_excitations, efficiency=0.96,
                      correct_B1=False, B1_map=None, B1_scale=1.0,
                      scale_phase=True,
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
        List of {first, second} inversion times, in seconds
    flip_angles: [float]
        List of {first, second} flip angles, in degrees
    inversion_TR: float
        Inversion repetition time, in seconds
    excitation_TR: [float]
        List of {first,second} repetition times, in seconds
    N_excitations: int
        Number of excitations
    efficiency: float
        Inversion efficiency (default is 0.96)
    correct_B1: bool
        Whether to correct for B1 inhomogeneities (default is False)
    B1_map: niimg
        Computed B1 map
    B1_scale: float
        B1 map scaling factor (default is 1.0)
    scale_phase: bool
        Whether to rescale the phase image in [0,2PI] or to assume it is 
        already in radians
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

        * t1 (niimg): Map of estimated T1 times in seconds (_qt1map-t1)
        * r1 (niimg): Map of estimated R1 relaxation rate in hertz (_qt1map-r1)
        * uni (niimg): Estimated T1 weighted image at TE=0 (_qt1map-uni)
        
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
        output_dir = _output_dir_4saving(output_dir, first_inversion[0])

        t1_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=first_inversion[0],
                                   suffix='qt1map-t1'))

        r1_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=first_inversion[0],
                                   suffix='qt1map-r1'))

        uni_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=first_inversion[0],
                                   suffix='qt1map-uni'))

        if overwrite is False \
            and os.path.isfile(t1_file) \
            and os.path.isfile(r1_file) \
            and os.path.isfile(uni_file) :
                output = {'t1': t1_file,
                          'r1': r1_file, 
                          'uni': uni_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    qt1map = nighresjava.IntensityMp2rageT1Fitting()

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
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    qt1map.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    qt1map.setResolutions(resolution[0], resolution[1], resolution[2])

    # input images
    qt1map.setFirstInversionMagnitude(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    data = load_volume(first_inversion[1]).get_data()
    qt1map.setFirstInversionPhase(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    data = load_volume(second_inversion[0]).get_data()
    qt1map.setSecondInversionMagnitude(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    data = load_volume(second_inversion[1]).get_data()
    qt1map.setSecondInversionPhase(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
 
    if (correct_B1):
        data = load_volume(B1_map).get_data()
        qt1map.setB1mapImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
        qt1map.setB1mapScaling(B1_scale)
 
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
    t1_data = np.reshape(np.array(qt1map.getQuantitativeT1mapImage(),
                                    dtype=np.float32), dimensions, 'F')

    r1_data = np.reshape(np.array(qt1map.getQuantitativeR1mapImage(),
                                    dtype=np.float32), dimensions, 'F')

    uni_data = np.reshape(np.array(qt1map.getUniformT1weightedImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(t1_data)
    header['cal_max'] = np.nanmax(t1_data)
    t1 = nb.Nifti1Image(t1_data, affine, header)

    header['cal_min'] = np.nanmin(r1_data)
    header['cal_max'] = np.nanmax(r1_data)
    r1 = nb.Nifti1Image(r1_data, affine, header)

    header['cal_min'] = np.nanmin(uni_data)
    header['cal_max'] = np.nanmax(uni_data)
    uni = nb.Nifti1Image(uni_data, affine, header)

    if save_data:
        save_volume(t1_file, t1)
        save_volume(r1_file, r1)
        save_volume(uni_file, uni)
        return {'t1': t1_file, 'r1': r1_file, 'uni': uni_file}
    else:
        return {'t1': t1, 'r1': r1, 'uni': uni}

def mp2rage_t1_from_uni(uniform_image, 
                      inversion_times, flip_angles, inversion_TR,
                      excitation_TR, N_excitations, efficiency=0.96,
                      correct_B1=False, B1_map=None, B1_scale=1.0,
                      scale_phase=True,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ MP2RAGE uniform image to T1 mapping

    Estimate T1/R1 by a look-up table method adapted from [1]_

    Parameters
    ----------
    uniform_image: niimg
        Uniform image computed from first and second inversion
    inversion_times: [float]
        List of {first, second} inversion times, in seconds
    flip_angles: [float]
        List of {first, second} flip angles, in degrees
    inversion_TR: float
        Inversion repetition time, in seconds
    excitation_TR: [float]
        List of {first,second} repetition times,in seconds
    N_excitations: int
        Number of excitations
    efficiency: float
        Inversion efficiency (default is 0.96)
    correct_B1: bool
        Whether to correct for B1 inhomogeneities (default is False)
    B1_map: niimg
        Computed B1 map
    B1_scale: float
        B1 map scaling factor (default is 1.0)
    scale_phase: bool
        Whether to rescale the phase image in [0,2PI] or to assume it is 
        already in radians
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

        * t1 (niimg): Map of estimated T1 times in seconds (_qt1map-t1)
        * r1 (niimg): Map of estimated R1 relaxation rate in hertz (_qt1map-r1)
        
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
        output_dir = _output_dir_4saving(output_dir, uniform_image)

        t1_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=uniform_image,
                                   suffix='qt1map-t1'))

        r1_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=uniform_image,
                                   suffix='qt1map-r1'))

        if overwrite is False \
            and os.path.isfile(t1_file) \
            and os.path.isfile(r1_file) :
                output = {'t1': t1_file,
                          'r1': r1_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    qt1map = nighresjava.IntensityMp2rageT1Fitting()

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
    img = load_volume(uniform_image)
    data = img.get_data()
    #data = data[0:10,0:10,0:10]
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    qt1map.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    qt1map.setResolutions(resolution[0], resolution[1], resolution[2])

    # input images
    qt1map.setUniformImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
     
    if (correct_B1):
        data = load_volume(B1_map).get_data()
        qt1map.setB1mapImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
        qt1map.setB1mapScaling(B1_scale)
 
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
    t1_data = np.reshape(np.array(qt1map.getQuantitativeT1mapImage(),
                                    dtype=np.float32), dimensions, 'F')

    r1_data = np.reshape(np.array(qt1map.getQuantitativeR1mapImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(t1_data)
    header['cal_max'] = np.nanmax(t1_data)
    t1 = nb.Nifti1Image(t1_data, affine, header)

    header['cal_min'] = np.nanmin(r1_data)
    header['cal_max'] = np.nanmax(r1_data)
    r1 = nb.Nifti1Image(r1_data, affine, header)

    if save_data:
        save_volume(t1_file, t1)
        save_volume(r1_file, r1)
        return {'t1': t1_file, 'r1': r1_file}
    else:
        return {'t1': t1, 'r1': r1}
