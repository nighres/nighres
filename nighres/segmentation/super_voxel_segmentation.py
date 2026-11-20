import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def super_voxel_segmentation(image, prior_seg=None, prior_proba=None, mask=None, scaling=4.0, noise_level=0.1, 
                      iterations=10, diff=0.01, threshold=0.5,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Super Voxel Segmentation

    Refines a segmentation result using a parcellation the image into regularly spaced super-voxels 
    of regular size and shape that follow intensity boundaries, based on Simple Non-iterative Clustering [1]_.
    The refinement step aims to compensate for local inhomogeneities, favoring continuity across parcels.

    Parameters
    ----------
    image: niimg
        Input image for the supervoxel parcellation
    prior_seg: niimg
        Input segmentation image to adjust
    prior_proba: niimg
        Input maximum probability image to adjust
    mask: niimg, optional
        Data mask to specify acceptable seeding regions
    scaling: float, optional
        Scaling factor for the new super-voxel grid (default is 4)
    noise_level: float, optional
        Weighting parameter to balance image intensity and spatial variability
    iterations: int, optional
        Maximum number of iterations in the segmentation adjustment step (default is 10)
    diff: float, optional
        Maximum difference in probabilities between steps before stopping (default is 0.01)
    threshold: float, optional
        Maximum boundary probability threshold (default is 0.5)
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
        * segmentation (niimg): The adjusted segmentation
        * posterior (niimg): The adjusted maximum probability posterior
    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    
    References
    ----------
    .. [1] R. Achanta and S. Suesstrunk, 
        Superpixels and Polygons using Simple Non-Iterative Clustering,
        Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2017.

    """

    print('\nSuper voxel segmentation')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        parcel_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='svs-parcel'))

        seg_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='svs-seg'))

        mems_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='svs-maxp'))

        if overwrite is False \
            and os.path.isfile(parcel_file) \
            and os.path.isfile(seg_file) \
            and os.path.isfile(mems_file) :
                print("skip computation (use existing results)")
                output = {'parcel': parcel_file, 'segmentation': seg_file, 'posterior': mems_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    supervoxel = nighresjava.SuperVoxelSegmentation()

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
    
    if prior_seg is not None:
        supervoxel.setPriorSegmentationImage(nighresjava.JArray('int')(
                (load_volume(prior_seg).get_fdata().flatten('F')).astype(int).tolist()))
    
    if type(prior_proba) is list:
        nseg = len(prior_proba)
        supervoxel.setPriorSegmentationNumber(nseg)
        for idx in range(nseg):
            supervoxel.setPriorImageAt(idx,nighresjava.JArray('float')(
                (load_volume(prior_proba[idx]).get_fdata().flatten('F')).astype(float)))
    elif prior_proba is not None:
        prior = load_volume(prior_proba).get_fdata()
        if len(prior.shape)==4:
            nseg = prior.shape[3]
            supervoxel.setPriorSegmentationNumber(nseg)
            for idx in range(nseg):
                supervoxel.setPriorImageAt(idx,nighresjava.JArray('float')(
                    (prior[:,:,:,idx].flatten('F')).astype(float)))
        else:
            supervoxel.setMaxPriorImage(nighresjava.JArray('float')(
                (prior.flatten('F')).astype(float)))
    
    if prior_seg is None and prior_proba is None:
        print("some prior segmentation information is needed")
        return
    
    if mask is not None:
        supervoxel.setMaskImage(nighresjava.JArray('int')(
                (load_volume(mask).get_fdata().flatten('F')).astype(int).tolist()))
    
    # set algorithm parameters
    supervoxel.setScalingFactor(scaling)
    supervoxel.setNoiseLevel(noise_level)
    supervoxel.setMaxIterations(iterations)
    supervoxel.setMaxDifference(diff)
    supervoxel.setThreshold(threshold)
    
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
                                    dtype=np.float32), shape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(parcel_data)
    header['cal_max'] = np.nanmax(parcel_data)
    parcel = nb.Nifti1Image(parcel_data, affine, header)

    #dims = supervoxel.getScaledDims()
    seg_data = np.reshape(np.array(supervoxel.getSegmentationImage(),
                                    dtype=np.int32), shape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(seg_data)
    header['cal_max'] = np.nanmax(seg_data)
    seg = nb.Nifti1Image(seg_data, affine, header)

    mems_data = np.reshape(np.array(supervoxel.getMaxPosteriorImage(),
                                    dtype=np.float32), shape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(mems_data)
    header['cal_max'] = np.nanmax(mems_data)
    mems = nb.Nifti1Image(mems_data, affine, header)

    if save_data:
        save_volume(parcel_file, parcel)
        save_volume(seg_file, seg)
        save_volume(mems_file, mems)
        return {'parcel': parcel_file, 'segmentation': seg_file, 'posterior': mems_file}
    else:
        return {'parcel': parcel, 'segmentation': seg, 'posterior': mems}
