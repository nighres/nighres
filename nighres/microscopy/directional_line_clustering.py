import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def directional_line_clustering(labels, scales, directions, thickness,
                            distance=20.0, angle=15.0, probability=0.5, anisotropy=0.0,
                            relabel=True, voxels=False, across=False, mip=0,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):
    """ Directional line clustering

    Uses a simple probabilistic model to cluster detected lines (from RRF module)
    in 2D images

    Parameters
    ----------
    labels: [niimg]
        Input 2D images with labels for the detected lines
    scales: [niimg]
        Input 2D images with the detection scale parameter 
    directions: [niimg]
        Input 3D images with the (x,y) detection direction 
    thickness: float
        Thickness in voxels between 2D slices
    distance: float
        Distance in voxels to expect between associated lines
    angle: float
        Angle in degrees  to expect between associated lines
    probability: float
        Probability threshold to use in line grouping
    anisotropy: float
        Anisotropy of the distance function relative to the lines direction
    relabel: bool
        Relabel input label image into separate 8/26-connected components (default is True)
    voxels: bool
        Use individual voxels from input image as components (default is False)
    across: bool
        Only group across images, not inside each one (default is False)
    mip: int
        Compute a maximum intensity projection of the grouping (default is 0)
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

        * direction (niimg): The estimated 3D direction for each detected line
        * grouping (niimg): The estimated grouping of the lines

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    
    """

    print('\nDirectional Line Clustering')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, labels[0])

        dir_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=labels[0],
                                   suffix='dlc-dir'))

        grp_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=labels[0],
                                   suffix='dlc-grp'))

        if overwrite is False \
            and os.path.isfile(dir_file) and os.path.isfile(grp_file) :
                print("skip computation (use existing results)")
                output =  {'direction': dir_file, 'grouping': grp_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    dlc = nighresjava.DirectionalLineClustering()

    # set parameters
    
    nimg = len(labels)
    dlc.setImageNumber(nimg)
    
    # load image and use it to set dimensions and resolution
    img = load_volume(labels[0])
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    if (len(dimensions)==2 or dimensions[2]==1):
        print("2D version")
        dlc.setDimensions(dimensions[0], dimensions[1], 1)
        dim2d = (dimensions[0], dimensions[1], nimg)
        dim3d = (dimensions[0], dimensions[1], 3*nimg)
    else:
        print("3D version")
        dlc.setDimensions(dimensions[0], dimensions[1], dimensions[2])
        dim2d = (dimensions[0], dimensions[1], dimensions[2], nimg)
        dim3d = (dimensions[0], dimensions[1], dimensions[2], 3*nimg)       
    
    for idx,label in enumerate(labels):
        data = load_volume(label).get_fdata()
        dlc.setLineImageAt(idx, nighresjava.JArray('int')(
                                (data.flatten('F')).astype(int).tolist()))
    
        data = load_volume(scales[idx]).get_fdata()
        dlc.setScaleImageAt(idx,nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
        data = load_volume(directions[idx]).get_fdata()
        dlc.setDirImageAt(idx,nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

    # set algorithm parameters
    dlc.setSliceThickness(thickness)
    dlc.setExpectedDistance(distance)
    dlc.setExpectedAngle(angle)
    dlc.setProbabilityThreshold(probability)
    dlc.setDistanceAnisotropy(anisotropy)
    dlc.setRecomputeLabels(relabel)
    dlc.setMaxIntensityProjection(mip)
    
    # execute the algorithm
    try:
        if (len(dimensions)==2 or dimensions[2]==1):
            dlc.execute2D()
        else:
            if voxels:
                print('building lines out of voxels')
                dlc.buildLines3D()
            elif across:
                print('combining lines from images')
                dlc.combineAcrossDimensions3D()
            else:
                print('grouping lines in all images')
                dlc.execute3D()
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    data = np.reshape(np.array(dlc.getDirectionImage(),
                                    dtype=np.float32), dim3d, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(data)
    header['cal_max'] = np.nanmax(data)
    dir_res = nb.Nifti1Image(data, affine, header)

    # reshape output to what nibabel likes
    data = np.reshape(np.array(dlc.getGroupImage(),
                                    dtype=np.int32), dim2d, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(data)
    header['cal_max'] = np.nanmax(data)
    grp_res = nb.Nifti1Image(data, affine, header)

    if save_data:
        save_volume(dir_file, dir_res)
        save_volume(grp_file, grp_res)
        return {'direction': dir_file, 'grouping': grp_file}
    else:
        return {'direction': dir_res, 'grouping': grp_res}
