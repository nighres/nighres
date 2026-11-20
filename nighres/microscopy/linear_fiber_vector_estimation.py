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


def linear_fiber_vector_estimation(proba_image, theta_image, lambda_image,
                              prior_image, mask_image=None,
                              orig_dim=7, prior_dim=3, vec_dim=3,
                              neighbors=3,search_radius=2,iterations=0,
                              init_threshold=0.25,thickness=15.0,offset=5.0,
                              delta_space=1.0, delta_depth=3.0, 
                              delta_theta=5.0, delta_prior=10.0,
                              save_data=False, overwrite=False, output_dir=None,
                              file_name=None):

    """ Linear Fiber Vector Estimation 

    Combines 2D lines extracted from stain images and 3D direction priors
    to estimate 3D directions in a registered microscopy stack

    Parameters
    ----------
    proba_image: niimg
        Input image providing 2D line detection probabilities.
    theta_image: niimg
        Input image providing 2D line directions.
    lambda_image: niimg
        Input image providing 2D line length.
    prior_image: niimg
        Input vector image providing prior directions (the 4th dimension is an array of
        N x (x,y,z) vectors.
    mask_image: niimg
        Input mask for the computation (only performed in voxels>0)
    orig_dim: int
        Number of original vectors per voxel (default is 7)
    prior_dim: int
        Number of prior vectors per voxel (default is 3)
    vec_dim: int
        Number of output vectors per voxel (default is 3)
    neighbors: int
        Number of neighboring z slices to average on both sides (default is 3)
    search_radius: int
        Distance in voxels to look for best neighbors in each slice (default is 2)
    iterations: int
        Number of iterations of the diffusion step (default is 0)
    init_threshold: float
        Probability threshold for using prior (default is 0.25)
    thickness: float
        Image visible thickness for depth computation (default is 15.0)
    offset: float
        Image visible thickness  offsetfor depth computation (default is 5.0)
    delta_space: float
        Scaling factor for spatial distances in 2D images (default is 1.0 voxel)
    delta_depth: float
        Scaling factor for spatial distances across the stack depth (default is 3.0 slices)
    delta_theta: float
        Scaling factor for angular differences in 2D (default is 5.0 degrees)
    delta_prior: float
        Scaling factor for angular differences in 3D (default is 10.0 degrees)
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

        * result (niimg): estimated vector map (_lfe-vec)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    References
    ----------

    """

    print('\n Linear Fiber Vector Estimation')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, proba_image)

        vec_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=proba_image,
                                  suffix='lfe-vec'))

        if overwrite is False \
            and os.path.isfile(vec_file):

            print("skip computation (use existing results)")
            output = {'result': vec_file}
            return output


    # load input image and use it to set dimensions and resolution
    img = load_volume(prior_image)
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = header.get_data_shape()
    dim4d = (dimensions[0],dimensions[1],dimensions[2],2)
    del img

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create extraction instance
    lfve = nighresjava.LinearFiberVectorEstimation()

    # set parameters
    lfve.setInputNumber(orig_dim)
    lfve.setPriorNumber(prior_dim)
    lfve.setVectorNumber(vec_dim)
    
    lfve.setNumberOfNeighbors(neighbors)
    lfve.setSearchRadius(search_radius)
    lfve.setIterations(iterations)
    
    lfve.setInitialThreshold(init_threshold)
    lfve.setImageThickness(thickness)
    lfve.setThicknessOffset(offset)
    
    lfve.setSpatialScale(delta_space)
    lfve.setDepthScale(delta_depth)
    lfve.setThetaScale(delta_theta)
    lfve.setPriorScale(delta_prior)
    
    lfve.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    lfve.setResolutions(resolution[0], resolution[1], resolution[2])

    # masking first: so we directly rescale?
    if (mask_image is not None):
        mask = (load_volume(mask_image).get_fdata()>0)
        lfve.setComputationMask(nighresjava.JArray('int')(
                                (mask.flatten('F')).astype(int).tolist()))
        
    # input image: generally too big to do in one go, so...
    print('load input files')
    data = load_volume(proba_image).get_fdata()   
    for n in range(orig_dim):
        print('.',end='')
        lfve.setProbaImageAt(n, nighresjava.JArray('float')(
                                            (data[:,:,:,n].flatten('F')).astype(float)))
    del data   
    print('proba')
    
    data = load_volume(theta_image).get_fdata()   
    for n in range(orig_dim):
        print('.',end='')
        lfve.setThetaImageAt(n, nighresjava.JArray('float')(
                                            (data[:,:,:,n].flatten('F')).astype(float)))
    del data   
    print('theta')
        
    data = load_volume(lambda_image).get_fdata()   
    for n in range(orig_dim):
        print('.',end='')
        lfve.setLambdaImageAt(n, nighresjava.JArray('float')(
                                            (data[:,:,:,n].flatten('F')).astype(float)))
    del data   
    print('lambda')
            
    # prior data
    data = load_volume(prior_image).get_fdata()   
    for n in range(3*prior_dim):
        print('.',end='')
        lfve.setPriorImageAt(n, nighresjava.JArray('float')(
                                            (data[:,:,:,n].flatten('F')).astype(float)))
    del data
    print('prior')
    
    # execute Extraction
    print('compute estimates...')
    try:
        lfve.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # new dimensions
    dimensions = (dimensions[0],dimensions[1],dimensions[2],3*vec_dim)
    dim_space = (dimensions[0],dimensions[1],dimensions[2])

    # reshape output to what nibabel likes
    vec_data = numpy.zeros(dimensions)
    print('extract output file...')
    for n in range(3*vec_dim):
        vec_data[:,:,:,n] = numpy.reshape(numpy.array(lfve.getVectorImageAt(n),
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
