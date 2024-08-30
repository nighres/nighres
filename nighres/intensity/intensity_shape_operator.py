import os
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving


def intensity_shape_operator(image, 
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):

    """Intensity shape operator

    Estimates a 3D shape operator tensor from an intensity image

    Parameters
    ----------
    image: niimg
        Intensity image to be processed
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

        * result (niimg): Tensor shape operator, in [xx,yy,zz,xy,xz,yz] ordering
            (output file suffix _ishop-tensor)

    Notes
    ----------
    Ported from original Java module by Pierre-Louis Bazin
    """

    print("\Intensity Shape Operator")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        tensor_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=image,
                                       suffix='ishop-tensor'))

        if overwrite is False \
            and os.path.isfile(tensor_file) :

            print("skip computation (use existing results)")
            output = {'result': tensor_file}
            return output

    # load the data
    img = load_volume(image)
    data = img.get_fdata()
    hdr = img.header
    aff = img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = data.shape
    dims4d = [dimensions[0],dimensions[1],dimensions[2],6]

    # algorithm
    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    algorithm = nighresjava.IntensityShapeOperator()

    # set parameters
    
    
    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])

    algorithm.setInputImage(nighresjava.JArray('float')(
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
    tensor_data = np.reshape(np.array(
                                    algorithm.getTensorImage(),
                                    dtype=np.float32), shape=dims4d, order='F')

    hdr['cal_min'] = np.nanmin(tensor_data)
    hdr['cal_max'] = np.nanmax(tensor_data)
    tensor = nb.Nifti1Image(tensor_data, aff, hdr)

    if save_data:
        save_volume(tensor_file, tensor)
        return {'result': tensor_file}
    else:
        return {'result': tensor}
