import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def t2s_optimal_combination(image_list, te_list, depth=None,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ T2*-Optimal Combination

    Estimate T2*/R2* and use it to combine multi-echo data

    Parameters
    ----------
    image_list: [niimg]
        List of 4D input images, one per echo time
    te_list: [float]
        List of input echo times (TE)
    depth: [int]
        List of echo depth to keep for input time points 
    save_data: bool, optional
        Save output data to file (default is False)
    overwrite: bool, optional
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

        * combined (niimg): Optimally combined 4D image
        * t2s (niimg): Map of estimated T2* times (_qt2fit-t2s)
        * r2s (niimg): Map of estimated R2* relaxation rate (_qt2fit-r2s)
        * s0 (niimg): Estimated PD weighted image at TE=0 (_qt2fit-s0)
        * residuals (niimg): Estimated residuals between input and estimated echoes (_qt2fit-err)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    print('\nT2* Optimal Combination')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image_list[0])

        comb_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image_list[0],
                                   suffix='qt2scomb-combined'))

        t2s_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image_list[0],
                                   suffix='qt2scomb-t2s'))

        r2s_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image_list[0],
                                   suffix='qt2scomb-r2s'))

        s0_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image_list[0],
                                   suffix='qt2scomb-s0'))

        err_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image_list[0],
                                   suffix='qt2scomb-err'))

        if overwrite is False \
            and os.path.isfile(comb_file) \
            and os.path.isfile(t2s_file) \
            and os.path.isfile(r2s_file) \
            and os.path.isfile(s0_file) \
            and os.path.isfile(err_file) :
                output = {'combined': comb_file,
                          't2s': t2s_file,
                          'r2s': r2s_file,
                          's0': s0_file,
                          'residuals': err_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    qt2scomb = nighresjava.T2sOptimalCombination()

    # set algorithm parameters
    qt2scomb.setNumberOfEchoes(len(image_list))

    # load first image and use it to set dimensions and resolution
    img = load_volume(image_list[0])
    data = img.get_data()
    #data = data[0:10,0:10,0:10]
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    dim3d = (dimensions[0], dimensions[1], dimensions[2])

    if len(dimensions)==3:
        qt2scomb.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    else:
        qt2scomb.setDimensions(dimensions[0], dimensions[1], dimensions[2], dimensions[3])
    qt2scomb.setResolutions(resolution[0], resolution[1], resolution[2])

    # input images
    # important: set image number before adding images
    for idx, image in enumerate(image_list):
        #print('\nloading ('+str(idx)+'): '+image)
        data = load_volume(image).get_data()
        #data = data[0:10,0:10,0:10]
        qt2scomb.setEchoImageAt(idx, nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

        qt2scomb.setEchoTimeAt(idx, te_list[idx])

    if depth is not None:
        qt2scomb.setImageEchoDepth(nighresjava.JArray('int')(depth))

    # execute the algorithm
    try:
        qt2scomb.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    comb_data = np.reshape(np.array(qt2scomb.getCombinedImage(),
                                    dtype=np.float32), dimensions, 'F')

    t2s_data = np.reshape(np.array(qt2scomb.getT2sImage(),
                                    dtype=np.float32), dim3d, 'F')

    r2s_data = np.reshape(np.array(qt2scomb.getR2sImage(),
                                    dtype=np.float32), dim3d, 'F')

    s0_data = np.reshape(np.array(qt2scomb.getS0Image(),
                                    dtype=np.float32), dim3d, 'F')

    err_data = np.reshape(np.array(qt2scomb.getResidualImage(),
                                    dtype=np.float32), dim3d, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(comb_data)
    header['cal_max'] = np.nanmax(comb_data)
    comb = nb.Nifti1Image(comb_data, affine, header)

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
        save_volume(comb_file, comb)
        save_volume(t2s_file, t2s)
        save_volume(r2s_file, r2s)
        save_volume(s0_file, s0)
        save_volume(err_file, err)

        return {'combined': comb_file, 't2s': t2s_file, 'r2s': r2s_file, 's0': s0_file, 'residuals': err_file}
    else:
        return {'combined': comb, 't2s': t2s, 'r2s': r2s, 's0': s0, 'residuals': err}
