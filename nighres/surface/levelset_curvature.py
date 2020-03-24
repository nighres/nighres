import os
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving


def levelset_curvature(levelset_image, distance=1.0,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):

    """Levelset curvature

    Estimates surface curvature of a levelset using a quadric approximation scheme.

    Parameters
    ----------
    levelset_image: niimg
        Levelset image to be turned into probabilities
    distance: float, optional
        Distance from the boundary in voxels where to estimate the curvature
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

        * mcurv (niimg): Mean curvature (output file suffix _curv-mean)
        * gcurv (niimg): Gaussian curvature (output file suffix _curv-gauss)

    Notes
    ----------
    Ported from original Java module by Pierre-Louis Bazin
    """

    print("\nLevelset Curvature")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, levelset_image)

        mcurv_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=levelset_image,
                                       suffix='curv-mean'))

        gcurv_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=levelset_image,
                                       suffix='curv-gauss'))

        if overwrite is False \
            and os.path.isfile(mcurv_file) \
            and os.path.isfile(gcurv_file) :

            print("skip computation (use existing results)")
            output = {'mcurv': mcurv_file, 'gcurv': gcurv_file}
            return output

    # load the data
    lvl_img = load_volume(levelset_image)
    lvl_data = lvl_img.get_data()
    hdr = lvl_img.header
    aff = lvl_img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = lvl_data.shape

    # algorithm
    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    algorithm = nighresjava.LevelsetCurvature()

    # set parameters
    algorithm.setMaxDistance(distance)

    # load images and set dimensions and resolution
    input_image = load_volume(levelset_image)
    data = input_image.get_data()
    affine = input_image.get_affine()
    header = input_image.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = input_image.shape

    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])

    algorithm.setLevelsetImage(nighresjava.JArray('float')(
                               (data.flatten('F')).astype(float)))

    # execute
    try:
        algorithm.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # Collect output
    mcurv_data = np.reshape(np.array(
                                    algorithm.getMeanCurvatureImage(),
                                    dtype=np.float32), dimensions, 'F')
    gcurv_data = np.reshape(np.array(
                                    algorithm.getGaussCurvatureImage(),
                                    dtype=np.float32), dimensions, 'F')

    hdr['cal_min'] = np.nanmin(mcurv_data)
    hdr['cal_max'] = np.nanmax(mcurv_data)
    mcurv = nb.Nifti1Image(mcurv_data, aff, hdr)

    hdr['cal_min'] = np.nanmin(gcurv_data)
    hdr['cal_max'] = np.nanmax(gcurv_data)
    gcurv = nb.Nifti1Image(gcurv_data, aff, hdr)

    if save_data:
        save_volume(mcurv_file, mcurv)
        save_volume(gcurv_file, gcurv)
        return {'mcurv': mcurv_file, 'gcurv': gcurv_file}
    else:
        return {'mcurv': mcurv, 'gcurv': gcurv}
