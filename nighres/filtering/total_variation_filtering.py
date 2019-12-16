import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def total_variation_filtering(image, mask=None, lambda_scale=0.05,
                      tau_step=0.125,max_dist=1e-4,max_iter=500,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Total Variation Filtering

    Total variation filtering.

    Parameters
    ----------
    image: niimg
        Input image to filter
    mask: niimg, optional
        Data mask for processing
    lambda_scale: float, optional
        Relative intensity scale for total variation smoothing (default is 0.5)
    tau_step: float, optional
        Internal step parameter (default is 0.125)
    max_dist: float, optional
        Maximum distance for convergence (default is 1e-4)
    max_iter: int, optional
        Maximum number of iterations (default is 500)
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

        * filtered (niimg): The filtered image (_tv-img)
        * residual (niimg): The image residuals (_tv-res)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Algorithm adapted from [1]_

    References
    ----------
    .. [1] Chambolle (2004). An Algorithm for Total Variation Minimization and
        Applications. doi:10.1023/B:JMIV.0000011325.36760.1e
    """

    print('\nTotal variation filtering')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        out_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='tv-img'))

        res_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='tv-res'))

        if overwrite is False \
            and os.path.isfile(out_file) and os.path.isfile(res_file) :
                print("skip computation (use existing results)")
                output = {'filtered': out_file,
                          'residual': res_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    algo = nighresjava.TotalVariationFiltering()

    # set parameters

    # load image and use it to set dimensions and resolution
    img = load_volume(image)
    data = img.get_data()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    algo.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    algo.setResolutions(resolution[0], resolution[1], resolution[2])

    algo.setImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))


    if mask is not None:
        algo.setMaskImage(idx, nighresjava.JArray('int')(
                (load_volume(mask).get_data().flatten('F')).astype(int).tolist()))

    # set algorithm parameters
    algo.setLambdaScale(lambda_scale)
    algo.setTauStep(tau_step)
    algo.setMaxDist(max_dist)
    algo.setMaxIter(max_iter)

    # execute the algorithm
    try:
        algo.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    filtered_data = np.reshape(np.array(algo.getFilteredImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(filtered_data)
    header['cal_max'] = np.nanmax(filtered_data)
    out = nb.Nifti1Image(filtered_data, affine, header)

    # reshape output to what nibabel likes
    residual_data = np.reshape(np.array(algo.getResidualImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(residual_data)
    header['cal_max'] = np.nanmax(residual_data)
    res = nb.Nifti1Image(residual_data, affine, header)

    if save_data:
        save_volume(out_file, out)
        save_volume(res_file, res)

        return {'filtered': out_file, 'residual': res_file}
    else:
        return {'filtered': out, 'residual': res}
