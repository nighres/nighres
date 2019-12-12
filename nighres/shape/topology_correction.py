import os
import sys
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def topology_correction(image, shape_type,
                    connectivity='wcs', propagation='object->background',
                    minimum_distance=0.00001, topology_lut_dir=None,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):

    """Topology correction

    Corrects the topology of a binary image, a probability map or a levelset
    surface to spherical with a fast marching technique [1]_.

    Parameters
    ----------
    image: niimg
        Image representing the shape of interest
    shape_type: {'binary_object','probability_map','signed_distance_function'}
        Which type of image is used as input
    connectivity: {'wcs','6/18','6/26','18/6','26/6'}
        What connectivity type to use (default is wcs: well-composed surfaces)
    propagation: {'object->background','background->object'}
        Which direction to use to enforce topology changes
        (default is 'object->background' )
    minimum_distance: float, optional
        Minimum distance to impose between successive voxels (default is 1e-5)
    topology_lut_dir: str, optional
        Path to directory in which topology files are stored (default is stored
        in TOPOLOGY_LUT_DIR)
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

        * corrected (niimg): Corrected image (output file suffix _tpc-img)
        * object (niimg): Corrected binary object (output file suffix _tpc-obj)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin

    References
    ----------
    .. [1] Bazin and Pham (2007). Topology correction of segmented medical
        images using a fast marching algorithm
        doi:10.1016/j.cmpb.2007.08.006
    """

    print("\nTopology Correction")

    # check topology_lut_dir and set default if not given
    topology_lut_dir = _check_topology_lut_dir(topology_lut_dir)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        corrected_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=image,
                                       suffix='tpc-img'))

        corrected_obj_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=image,
                                       suffix='tpc-obj'))
        if overwrite is False \
            and os.path.isfile(corrected_file) \
            and os.path.isfile(corrected_obj_file) :

            print("skip computation (use existing results)")
            output = {'corrected': corrected_file,
                      'object': corrected_obj_file}
            return output

    # start virtual machine if not running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initiate class
    algorithm = nighresjava.ShapeTopologyCorrection2()

    # load the data
    img = load_volume(image)
    hdr = img.header
    aff = img.affine
    data = img.get_data()
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = data.shape

    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])
    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])

    algorithm.setShapeImage(nighresjava.JArray('float')(
                            (data.flatten('F')).astype(float)))

    algorithm.setShapeImageType(shape_type)

    algorithm.setTopology(connectivity)
    algorithm.setTopologyLUTdirectory(topology_lut_dir)

    algorithm.setPropagationDirection(propagation)

    algorithm.setMinimumDistance(minimum_distance)

    # execute class
    try:
        algorithm.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # collect outputs
    corrected_data = np.reshape(np.array(algorithm.getCorrectedImage(),
                               dtype=np.float32), dimensions, 'F')

    hdr['cal_min'] = np.nanmin(corrected_data)
    hdr['cal_max'] = np.nanmax(corrected_data)
    corrected = nb.Nifti1Image(corrected_data, aff, hdr)

    corrected_obj_data = np.reshape(np.array(algorithm.getCorrectedObjectImage(),
                               dtype=np.int32), dimensions, 'F')

    hdr['cal_min'] = np.nanmin(corrected_obj_data)
    hdr['cal_max'] = np.nanmax(corrected_obj_data)
    corrected_obj = nb.Nifti1Image(corrected_obj_data, aff, hdr)

    if save_data:
        save_volume(corrected_file, corrected)
        save_volume(corrected_obj_file, corrected_obj)

        return {'corrected': corrected_file, 'object': corrected_obj_file}
    else:
        return {'corrected': corrected, 'object': corrected_obj}
