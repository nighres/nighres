import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def intensity_based_skullstripping(main_image, extra_image=None,
                            noise_model='exponential', skip_zero_values=True,
                            iterate=False, dilate_mask=0, dynamic_range=0.8,
                            topology_lut_dir=None,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):
    """ Intensity-based skull stripping

    Estimate a brain mask for a dataset with good brain/background intensity 
    separation (e.g. PD-weighted). An extra image can be used to ensure high 
    intensities are preserved (e.g. T1 map, T2-weighted data or a probability 
    map for a ROI).
		
    Parameters
    ----------
    main_image: niimg
        Main Intensity Image
    extra_image: niimg, optional
        Extra image with high intensity at brain boundary
    noise_model: {'exponential','half-normal','exp+log-normal','half+log-normal'}
        Background noise model (default is 'exponential')
    skip_zero_values: bool
        Ignores voxels with zero value (default is True)
    iterate: bool
        Whether to iterate the estimation (may be unstable in some cases, 
        default is False)
    dilate_mask: int
         Additional dilation (or erosion, if negative) of the brain mask 
         (default is 0)
    dynamic_range: float
         Dynamic range for the foreground / background differences in [0,1]
         (default is 0.8)
    topology_lut_dir: str, optional
        Path to directory in which topology files are stored (default is stored
        in TOPOLOGY_LUT_DIR)
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

        * brain_mask (niimg): Binary brain mask (_istrip-mask)
        * brain_proba (niimg): Probability brain map (_istrip-proba)
        * main_masked (niimg): Masked main image (_istrip-main)
        * extra_masked (niimg): Masked extra map (_istrip-extra)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Details on the algorithm can 
    be found in [1]_ 
    
    References
    ----------
    .. [1] Bazin et al. (2014). A computational framework for ultra-high 
       resolution cortical segmentation at 7 Tesla.
       DOI: 10.1016/j.neuroimage.2013.03.077
    """

    print('\nIntensity-based Skull Stripping')

    # check topology lut dir and set default if not given
    topology_lut_dir = _check_topology_lut_dir(topology_lut_dir)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, main_image)

        mask_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=main_image,
                                   suffix='istrip-mask'))
        proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=main_image,
                                   suffix='istrip-proba'))
        main_file = os.path.join(output_dir, 
                    _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=main_image,
                                  suffix='istrip-main'))

        if extra_image is not None:
            extra_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                        rootfile=extra_image,
                                        suffix='istrip-extra'))
        else:
            extra_file = None
        
        if overwrite is False \
            and os.path.isfile(mask_file) \
            and os.path.isfile(proba_file) \
            and os.path.isfile(main_file) :
            
            print("skip computation (use existing results)")
            output = {'brain_mask': mask_file, 
                    'brain_proba': proba_file, 
                    'main_masked': main_file}
            if extra_file is not None:
                if os.path.isfile(extra_file) :     
                    output['extra_masked'] = extra_file
            return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # create skulltripping instance
    algo = nighresjava.BrainIntensityBasedSkullStripping()

    # get dimensions and resolution from second inversion image
    main_img = load_volume(main_image)
    main_data = main_img.get_data()
    main_affine = main_img.affine
    main_hdr = main_img.header
    resolution = [x.item() for x in main_hdr.get_zooms()]
    dimensions = main_data.shape
    algo.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    algo.setResolutions(resolution[0], resolution[1], resolution[2])
    algo.setMainIntensityImage(nighresjava.JArray('float')(
                                    (main_data.flatten('F')).astype(float)))

    # pass other inputs
    if extra_image is not None:
        extra_img = load_volume(extra_image)
        extra_data = extra_img.get_data()
        extra_affine = extra_img.affine
        extra_hdr = extra_img.header
        algo.setExtraIntensityImage(nighresjava.JArray('float')(
                                    (extra_data.flatten('F')).astype(float)))

    algo.setBackgroundNoiseModel(noise_model)
    algo.setIterativeEstimation(iterate)
    algo.setSkipZeroValues(skip_zero_values)
    algo.setAdditionalMaskDilation(dilate_mask)
    algo.setTopologyLUTdirectory(topology_lut_dir)
    algo.setDynamicRange(dynamic_range)

    # execute skull stripping
    try:
        algo.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # collect outputs and potentially save
    main_masked_data = np.reshape(np.array(
                                algo.getMaskedMainImage(),
                                dtype=np.float32), dimensions, 'F')
    main_hdr['cal_max'] = np.nanmax(main_masked_data)
    main_masked = nb.Nifti1Image(main_masked_data, main_affine, main_hdr)

    mask_data = np.reshape(np.array(algo.getBrainMaskImage(),
                                    dtype=np.uint32), dimensions, 'F')
    main_hdr['cal_max'] = np.nanmax(mask_data)
    mask = nb.Nifti1Image(mask_data, main_affine, main_hdr)

    proba_data = np.reshape(np.array(
                                algo.getForegroundProbabilityImage(),
                                dtype=np.float32), dimensions, 'F')
    main_hdr['cal_max'] = np.nanmax(proba_data)
    proba = nb.Nifti1Image(proba_data, main_affine, main_hdr)

    if extra_image is not None:
        extra_data = np.reshape(np.array(
                                algo.getMaskedExtraImage(),
                                dtype=np.float32), dimensions, 'F')
        extra_hdr['cal_max'] = np.nanmax(extra_data)
        extra_masked = nb.Nifti1Image(extra_data, extra_affine, extra_hdr)

    if save_data:
        save_volume(main_file, main_masked)
        save_volume(mask_file, mask)
        save_volume(proba_file, proba)
        outputs = {'brain_mask': mask_file, 'brain_proba': proba_file, 'main_masked': main_file}
        if extra_image is not None:
            save_volume(extra_file, extra_masked)
            outputs['extra_masked'] = extra_file
    else:
        outputs = {'brain_mask': mask, 'brain_proba': proba, 'main_masked': main_masked}
        if extra_image is not None:
            outputs['extra_masked'] = extra_masked
            
    return outputs
