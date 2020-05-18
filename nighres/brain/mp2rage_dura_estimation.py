import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def mp2rage_dura_estimation(second_inversion, skullstrip_mask,
                           background_distance=5.0, output_type='dura_region',
                           save_data=False, overwrite=False, output_dir=None,
                           file_name=None):
    """ MP2RAGE dura estimation

    Filters a MP2RAGE brain image to obtain a probability map of dura matter.

    Parameters
    ----------
    second_inversion: niimg
        Second inversion image derived from MP2RAGE sequence
    skullstrip_mask: niimg
        Skullstripping mask defining the approximate region including the brain
    background_distance: float
        Maximum distance within the mask for dura (default is 5.0 mm)
    output_type: {'dura_region','boundary','dura_prior','bg_prior',
        'intens_prior'}
        Type of output result (default is 'dura_region')
    save_data: bool
        Save output data to file (default is False)
    overwrite: bool
        Overwrite existing results (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)
    return_filename: bool, optional
        Return filename instead of object

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * result (niimg): Dura probability image (_dura-proba)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Details on the algorithm can
    be found in [1]_ and a presentation of the MP2RAGE sequence in [2]_

    References
    ----------
    .. [1] Bazin et al. (2014). A computational framework for ultra-high
       resolution cortical segmentation at 7 Tesla.
       DOI: 10.1016/j.neuroimage.2013.03.077
    .. [2] Marques et al. (2010). MP2RAGE, a self bias-field corrected sequence
       for improved segmentation and T1-mapping at high field.
       DOI: 10.1016/j.neuroimage.2009.10.002
    """

    print('\nMP2RAGE Dura Estimation')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, second_inversion)

        result_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=second_inversion,
                                   suffix='dura-proba'))

        if overwrite is False \
            and os.path.isfile(result_file) :

            print("skip computation (use existing results)")
            output = {'result': result_file}
            return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # create skulltripping instance
    algo = nighresjava.BrainMp2rageDuraEstimation()

    # get dimensions and resolution from second inversion image
    inv2_img = load_volume(second_inversion)
    inv2_data = inv2_img.get_data()
    inv2_affine = inv2_img.affine
    inv2_hdr = inv2_img.header
    resolution = [x.item() for x in inv2_hdr.get_zooms()]
    dimensions = inv2_data.shape
    algo.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    algo.setResolutions(resolution[0], resolution[1], resolution[2])
    algo.setSecondInversionImage(nighresjava.JArray('float')(
                                    (inv2_data.flatten('F')).astype(float)))

    # pass other inputs
    mask_data = load_volume(skullstrip_mask).get_data()
    algo.setSkullStrippingMask(nighresjava.JArray('int')(
                                    (mask_data.flatten('F')).astype(int).tolist()))

    algo.setDistanceToBackground_mm(background_distance)
    algo.setOutputType(output_type)

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
    result_data = np.reshape(np.array(
                                algo.getDuraImage(),
                                dtype=np.float32), dimensions, 'F')
    inv2_hdr['cal_max'] = np.nanmax(result_data)
    result_img = nb.Nifti1Image(result_data, inv2_affine, inv2_hdr)

    if save_data:
        save_volume(result_file, result_img)
        outputs = {'result': result_file}
    else:
        outputs = {'result': result_img}
        
    return outputs
