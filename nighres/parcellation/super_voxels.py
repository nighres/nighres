import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def super_voxels(image, mask=None, scaling=4.0, noise_level=0.1, output_type='average',
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Super Voxels

    Parcellates the image into regularly spaced super-voxels of regular size and shape that
    follow intensity boundaries, based on Simple Non-iterative Clustering [1]_.

    Parameters
    ----------
    image: niimg
        Input image
    mask: niimg, optional
        Data mask to specify acceptable seeding regions
    scaling: float, optional
        Scaling factor for the new super-voxel grid (default is 4)
    noise_level: float, optional
        Weighting parameter to balance image intensity and spatial variability
    output_type: string, optional
        Output measures: "average", "difference" "distance" "sharpness" (default is "average")
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

        * parcel (niimg): The super-voxel parcellation of the original image
        * rescaled (niimg): The rescaled image of supervoxels
        * mems (niimg): The membership estimate of original image voxels to corresponding 
        super-voxels

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    
    References
    ----------
    .. [1] R. Achanta and S. Suesstrunk, 
        Superpixels and Polygons using Simple Non-Iterative Clustering,
        Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2017.

    """

    print('\nSuper voxels')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        parcel_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='svx-parcel'))

        rescaled_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='svx-rescale'))

        mems_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='svx-mems'))

        if overwrite is False \
            and os.path.isfile(parcel_file) \
            and os.path.isfile(rescaled_file) \
            and os.path.isfile(mems_file) :
                print("skip computation (use existing results)")
                output = {'parcel': parcel_file, 'rescaled': rescaled_file, 'mems': mems_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    supervoxel = nighresjava.SuperVoxels()

    # set parameters
    
    # load image and use it to set dimensions and resolution
    img = load_volume(image)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    if len(dimensions)>2:
        supervoxel.setDimensions(dimensions[0], dimensions[1], dimensions[2])
        supervoxel.setResolutions(resolution[0], resolution[1], resolution[2])
    else:
        supervoxel.setDimensions(dimensions[0], dimensions[1])
        supervoxel.setResolutions(resolution[0], resolution[1])
        
    supervoxel.setInputImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    
    if mask is not None:
        supervoxel.setMaskImage(nighresjava.JArray('int')(
                (load_volume(mask).get_fdata().flatten('F')).astype(int).tolist()))
    
    # set algorithm parameters
    supervoxel.setScalingFactor(scaling)
    supervoxel.setNoiseLevel(noise_level)
    supervoxel.setOutputType(output_type)
    
    # execute the algorithm
    try:
        supervoxel.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    parcel_data = np.reshape(np.array(supervoxel.getParcelImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(parcel_data)
    header['cal_max'] = np.nanmax(parcel_data)
    parcel = nb.Nifti1Image(parcel_data, affine, header)

    dims = supervoxel.getScaledDims()
    rescaled_data = np.reshape(np.array(supervoxel.getRescaledImage(),
                                    dtype=np.float32), dims, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(rescaled_data)
    header['cal_max'] = np.nanmax(rescaled_data)
    rescaled = nb.Nifti1Image(rescaled_data, affine, header)

    mems_data = np.reshape(np.array(supervoxel.getMemsImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(mems_data)
    header['cal_max'] = np.nanmax(mems_data)
    mems = nb.Nifti1Image(mems_data, affine, header)

    if save_data:
        save_volume(parcel_file, parcel)
        save_volume(rescaled_file, rescaled)
        save_volume(mems_file, mems)
        return {'parcel': parcel_file, 'rescaled': rescaled_file, 'mems': mems_file}
    else:
        return {'parcel': parcel, 'rescaled': rescaled, 'mems': mems}
