# basic dependencies
import os
import sys

# main dependencies: numpy, nibabel
import numpy as np
import nibabel as nb

# nighresjava and nighres functions
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
    _check_topology_lut_dir, _check_available_memory


def spectral_voxel_spatial_embedding(image, 
                    reference=None,
                    mapping=None,
                    dims=10,
                    msize=500,
                    scale=50.0,
                    space=50.0,
                    link=1.0,
                    depth=18,
                    alpha=0.0,
                    rotate=True,
                    affinity="linear",
                    save_data=False, 
                    overwrite=False, 
                    output_dir=None,
                    file_name=None):

    """ Spectral voxel spatial embedding
    
    Derive a spectral Laplacian embedding from a voxel domain

    Parameters
    ----------
    image: niimg
        Image of the structure of interest
    reference: niimg
        Image of the reference (optional)
    mapping: niimg
        Coordinate mapping from the image to the reference (optional)
    dims: int
        Number of kept dimensions in the representation (default is 1)
    msize: int
        Target matrix size for subsampling (default is 2000)
    scale: float
        Distance scale between sample points (default is 10.0)
    space: float
        Spatial scaling factor (default is 10.0)
    link: float
        Spatial linking factor (default is 1.0)
    depth: int
        Number of nearest neighbors used in first approximation (default is 18)
    alpha: float
        Laplacian norm parameter in [0:1] (default is 0.0)
    rotate: bool
        Rotate joint embeddings to match the reference (default is True)
    affinity: String
        Type of affinity kernel to use ({'linear', 'Cauchy', 'Gauss'}, default is 'linear')
    save_data: bool, optional
        Save output data to file (default is False)
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

        * result (niimg): Coordinate map (_sme-coord)

    Notes
    ----------
    
    References
    ----------

    """

    print("\nSpectral Voxel Spatial Embedding")

    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        coord_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=image,
                                  suffix='svse-coord'))
        
        if reference is not None:
            ref_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=image,
                                  suffix='svse-ref'))

        if overwrite is False \
            and os.path.isfile(coord_file) \
            and (reference is None or os.path.isfile(ref_file)) :
                print("skip computation (use existing results)")
                if reference is None:
                    output = {'result': coord_file}
                else:
                    output = {'result': coord_file, 'reference': ref_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    algorithm = nighresjava.SpectralVoxelEmbedding()

    # load the data
    image = load_volume(image)
    data = image.get_fdata()
    affine = image.affine
    header = image.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = image.shape
    dimensions4 = (dimensions[0],dimensions[1],dimensions[2],dims)
    
    algorithm.setImageDimensions(dimensions[0], dimensions[1], dimensions[2])
    algorithm.setImageResolutions(resolution[0], resolution[1], resolution[2])

    algorithm.setInputImage(nighresjava.JArray('float')(
                               (data.flatten('F')).astype(float)))

    if reference is not None:
        
        reference = load_volume(reference)
        resref = [x.item() for x in reference.header.get_zooms()]
        dimref = reference.shape
        dimref4 = (dimref[0],dimref[1],dimref[2],dims)
        
        algorithm.setReferenceDimensions(dimref[0], dimref[1], dimref[2])
        algorithm.setReferenceResolutions(resref[0], resref[1], resref[2])

        algorithm.setReferenceImage(nighresjava.JArray('float')(
                               (reference.get_fdata().flatten('F')).astype(float)))

        if mapping is not None:            
            mapping = load_volume(mapping)
            algorithm.setMapping(nighresjava.JArray('float')(
                               (mapping.get_fdata().flatten('F')).astype(float)))
            
    
    algorithm.setDimensions(dims)
    
    algorithm.setMatrixSize(msize)
    
    algorithm.setDistanceScale(scale)
    algorithm.setSpatialScale(space)    
    algorithm.setLinkingFactor(link)
    algorithm.setAffinityType(affinity)
    
    # execute
    try:
        if reference is not None:
            if reference is image:
                algorithm.voxelDistanceReferenceSparseEmbedding(depth,alpha)
            else:
                if rotate: 
                    algorithm.rotatedJointSpatialEmbedding(depth,alpha)
                else:
                    algorithm.voxelDistanceJointSparseEmbedding(depth,alpha)
        else:
            algorithm.voxelDistanceSparseEmbedding(depth,alpha)
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # Collect output
    coord_data = np.reshape(np.array(algorithm.getImageEmbedding(),
                               dtype=np.float32), newshape=dimensions4, order='F')
    
    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(coord_data)
    header['cal_max'] = np.nanmax(coord_data)
    coord_img = nb.Nifti1Image(coord_data, affine, header)

    if reference is not None:
        ref_data = np.reshape(np.array(algorithm.getReferenceEmbedding(),
                               dtype=np.float32), newshape=dimref4, order='F')

        header['cal_min'] = np.nanmin(ref_data)
        header['cal_max'] = np.nanmax(ref_data)
        ref_img = nb.Nifti1Image(ref_data, affine, header)
        

    if save_data:
        if reference is not None:
            save_volume(coord_file, coord_img)
            save_volume(ref_file, ref_img)
            return {'result': coord_file, 'reference': ref_file}
        else:
            save_volume(coord_file, coord_img)
            return {'result': coord_file}
    else:
        if reference is not None:
            return {'result': coord_img, 'reference': ref_img}
        else:
            return {'result': coord_img}


def spectral_voxel_data_embedding(image, 
                    reference=None,
                    mapping=None,
                    dims=10,
                    msize=500,
                    scale=50.0,
                    space=50.0,
                    link=1.0,
                    alpha=0.0,
                    rotate=True,
                    affinity="linear",
                    distance="product",
                    save_data=False, 
                    overwrite=False, 
                    output_dir=None,
                    file_name=None):

    """ Spectral voxel spatial embedding
    
    Derive a spectral Laplacian embedding from a voxel domain

    Parameters
    ----------
    image: niimg
        Image of the structure of interest
    reference: niimg
        Image of the reference (optional)
    mapping: niimg
        Coordinate mapping from the image to the reference (optional)
    dims: int
        Number of kept dimensions in the representation (default is 1)
    msize: int
        Target matrix size for subsampling (default is 2000)
    scale: float
        Distance scale between sample points (default is 10.0)
    space: float
        Spatial scaling factor (default is 10.0)
    link: float
        Spatial linking factor (default is 1.0)
    alpha: float
        Laplacian norm parameter in [0:1] (default is 0.0)
    rotate: bool
        Rotate joint embeddings to match the reference (default is True)
    affinity: String
        Type of affinity kernel to use ({'linear', 'Cauchy', 'Gauss'}, default is 'linear')
    distance: String
        Type of distance function to use ({'Euclidean', 'product', 'cosine'}, default is 'product')
    save_data: bool, optional
        Save output data to file (default is False)
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

        * result (niimg): Coordinate map (_sme-coord)

    Notes
    ----------
    
    References
    ----------

    """

    print("\nSpectral Voxel Data Embedding")

    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        coord_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=image,
                                  suffix='svde-coord'))
        
        if reference is not None:
            ref_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=image,
                                  suffix='svde-ref'))

        if overwrite is False \
            and os.path.isfile(coord_file) :
                print("skip computation (use existing results)")
                output = {'result': coord_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    algorithm = nighresjava.SpectralVoxelDataEmbedding()

    # load the data
    image = load_volume(image)
    data = image.get_fdata()
    affine = image.affine
    header = image.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = image.shape
    dimensions4 = (dimensions[0],dimensions[1],dimensions[2],dims)
    
    if len(dimensions)==3: 
        algorithm.setImageDimensions(dimensions[0], dimensions[1], dimensions[2], 1)
    else:
        algorithm.setImageDimensions(dimensions[0], dimensions[1], dimensions[2], dimensions[3])
    algorithm.setImageResolutions(resolution[0], resolution[1], resolution[2])

    algorithm.setInputImage(nighresjava.JArray('float')(
                               (data.flatten('F')).astype(float)))

    if reference is not None:
        
        reference = load_volume(reference)
        resref = [x.item() for x in reference.header.get_zooms()]
        dimref = reference.shape
        dimref4 = (dimref[0],dimref[1],dimref[2],dims)
        
        if len(dimref)==3: 
            algorithm.setReferenceDimensions(dimref[0], dimref[1], dimref[2], 1)
        else:
            algorithm.setReferenceDimensions(dimref[0], dimref[1], dimref[2], dimref[3])
        algorithm.setReferenceResolutions(resref[0], resref[1], resref[2])

        algorithm.setReferenceImage(nighresjava.JArray('float')(
                               (reference.get_fdata().flatten('F')).astype(float)))

        if mapping is not None:            
            mapping = load_volume(mapping)
            algorithm.setMapping(nighresjava.JArray('float')(
                               (mapping.get_fdata().flatten('F')).astype(float)))
            
    
    algorithm.setDimensions(dims)
    
    algorithm.setMatrixSize(msize)
    
    algorithm.setDistanceScale(scale)
    algorithm.setSpatialScale(space)    
    algorithm.setLinkingFactor(link)
    algorithm.setAffinityType(affinity)
    algorithm.setDistanceType(distance)
    
    # execute
    try:
        if reference is not None:
            if reference is image:
                algorithm.voxelDataReferenceSparseEmbedding(depth,alpha)
            else:
                if rotate: 
                    algorithm.rotatedJointDataEmbedding(depth,alpha)
                else:
                    algorithm.voxelDataJointSparseEmbedding(depth,alpha)
        else:
            algorithm.voxelDataSparseEmbedding(depth,alpha)
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # Collect output
    coord_data = np.reshape(np.array(algorithm.getImageEmbedding(),
                               dtype=np.float32), newshape=dimensions4, order='F')
    
    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(coord_data)
    header['cal_max'] = np.nanmax(coord_data)
    coord_img = nb.Nifti1Image(coord_data, affine, header)

    if reference is not None:
        ref_data = np.reshape(np.array(algorithm.getReferenceEmbedding(),
                               dtype=np.float32), newshape=dimref4, order='F')

        header['cal_min'] = np.nanmin(ref_data)
        header['cal_max'] = np.nanmax(ref_data)
        ref_img = nb.Nifti1Image(ref_data, affine, header)
        

    if save_data:
        if reference is not None:
            save_volume(coord_file, coord_img)
            save_volume(ref_file, ref_img)
            return {'result': coord_file, 'reference': ref_file}
        else:
            save_volume(coord_file, coord_img)
            return {'result': coord_file}
    else:
        if reference is not None:
            return {'result': coord_img, 'reference': ref_img}
        else:
            return {'result': coord_img}


