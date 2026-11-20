import numpy
import nibabel
import os
import sys
import math
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, \
                    _check_available_memory


def linear_fiber_vector_averaging(vector_image,
                              orig_dim=7, kept_dim=14,
                              neighbors=3,search_radius=2,
                              threshold=0.001,
                              save_data=False, overwrite=False, output_dir=None,
                              file_name=None):

    """ Linear Fiber Vector Averaging 

    Combines stacked direction vectors across z slices, using a non-local maximum
    combination approach

    Parameters
    ----------
    vector_image: niimg
        Input vector image providing directions (the 4th dimension is an array of
        2N x (x,y,z) vectors, with the second half mirrored from the first in the
        z direction; generated from a linear fiber model).
    orig_dim: int
        Number of original vectors per voxel (default is 7)
    kept_dim: int
        Number of kept vectors per voxel (default is 14)
    neighbors: int
        Number of neighboring z slices to average on both sides (default is 3)
    search_radius: int
        Distance in voxels to look for best neighbors in each slice (default is 2)
    threshold: float
        Minimum vector size to be taken into account (default is 0.001)
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

        * result (niimg): averaged vector map (_lfva-vec)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    References
    ----------

    """

    print('\n Linear Fiber Vector Averaging')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, vector_image)

        vec_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=vector_image,
                                  suffix='lfva-vec'))

        if overwrite is False \
            and os.path.isfile(vec_file):

            print("skip computation (use existing results)")
            output = {'result': vec_file}
            return output


    # load input image and use it to set dimensions and resolution
    img = load_volume(vector_image)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create extraction instance
    lfva = nighresjava.LinearFiberVectorAveraging()

    # set parameters
    lfva.setOriginalVectorNumber(orig_dim)
    lfva.setKeptVectorNumber(kept_dim)
    lfva.setNumberOfNeighbors(neighbors)
    lfva.setSearchRadius(search_radius)
    lfva.setVectorThreshold(threshold)
    
    lfva.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    lfva.setResolutions(resolution[0], resolution[1], resolution[2])

    # input image: generally too big to do in one go, so...
    print('load input file...')
    for n in range(3*orig_dim):
        lfva.setVectorImageAt(n, nighresjava.JArray('float')(
                                            (data[:,:,:,n].flatten('F')).astype(float)))

    # execute Extraction
    print('compute average...')
    try:
        lfva.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # new dimensions
    dimensions = (dimensions[0],dimensions[1],dimensions[2],6*kept_dim)
    dim_space = (dimensions[0],dimensions[1],dimensions[2])

    # reshape output to what nibabel likes
    vec_data = numpy.zeros(dimensions)
    print('extract output file...')
    for n in range(6*kept_dim):
        vec_data[:,:,:,n] = numpy.reshape(numpy.array(lfva.getAveragedVectorImageAt(n),
                                    dtype=numpy.float32), shape=dim_space, order='F')

 
    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = numpy.nanmax(vec_data)
    vec_img = nibabel.Nifti1Image(vec_data, affine, header)

    if save_data:
        save_volume(vec_file, vec_img)
       
        return {'result': vec_file}
    else:
        return {'result': vec_img}
