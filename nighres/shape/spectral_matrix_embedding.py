# basic dependencies
import os
import sys

# main dependencies: numpy, nibabel
import numpy as np
import nibabel as nb

# nighresjava and nighres functions
import nighresjava
from ..io import load_volume, save_volume, load_mesh_geometry, save_mesh_geometry, \
    load_mesh, save_mesh
from ..utils import _output_dir_4saving, _fname_4saving, \
    _check_topology_lut_dir, _check_available_memory


def spectral_matrix_embedding(distance_matrix, 
                    reference_matrix=None,
                    correspondence_matrix=None,
                    ref_correspondence_matrix=None,
                    surface_mesh=None,
                    reference_mesh=None,
                    dims=10,
                    msize=500,
                    scale=50.0,
                    space=50.0,
                    link=2.0,
                    normalize=True,
                    rotate=True,
                    save_data=False, 
                    overwrite=False, 
                    output_dir=None,
                    file_name=None):

    """ Spectral matrix embedding
    
    Derive a spectral Laplacian embedding from a distance matrix,
    optionally taking a reference mesh into account.
    
    Computations are sped up with a simplified Nyström approximation

    Parameters
    ----------
    distance_matrix: npy
        Full distance matrix (2D numpy array)
    reference_matrix: npy
        Reference distance matrix, optional (2D numpy array)
    correspondence_matrix: npy
        User-provided subject-to-reference correspondence distance matrix, optional (2D numpy array)
    ref_correspondence_matrix: npy
        User-provided reference-to-reference geodesic distance matrix, optional (2D numpy array)
    surface_mesh: mesh, optional
        Mesh model of a surface to define the geometry and display the results on.
    reference_mesh: mesh, optional
        Mesh model of a surface to define the reference geometry.
    dims: int
        Number of kept dimensions in the representation (default is 10)
    msize: int
        Target matrix size for subsampling (default is 500)
    scale: float
        Distance scaling factor (default is 50.0)
    space: float
        Spatial scaling factor (default is 50.0)
    link: float
        Spatial linking factor (default is 2.0)
    normalize: bool
        Normalizes embeddings to unit norm (default is True)
    rotate: bool
        Rotate joint embeddings to match the reference (default is True)
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

        * result (npy): Coordinate vectors (_smx-coord)

    Notes
    ----------
    
    References
    ----------

    """

    print("\nSpectral Matrix Embedding")

    if save_data:
        output_dir = _output_dir_4saving(output_dir, distance_matrix)

        matrix_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=distance_matrix,
                                  suffix='smx-coord',ext='npy'))

        if reference_matrix is not None:
            ref_matrix_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=distance_matrix,
                                  suffix='smx-ref',ext='npy'))

        if surface_mesh is not None:
            surf_matrix_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=distance_matrix,
                                  suffix='smx-surf',ext='vtk'))
            
        if reference_mesh is not None:
            surf_ref_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=distance_matrix,
                                  suffix='smx-refsurf',ext='vtk'))
            
        if overwrite is False \
            and os.path.isfile(matrix_file) :
                print("skip computation (use existing results)")
                output = {'result': matrix_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    algorithm = nighresjava.SpectralDenseMatrixEmbedding()

    # load the data
    if not isinstance(distance_matrix,np.ndarray): 
        distance_matrix = np.load(distance_matrix)
        
    algorithm.setSubjectMatrix(nighresjava.JArray('double')(
                            (distance_matrix.flatten('C')).astype(float)))
    
    if surface_mesh is not None:
        surface_mesh = load_mesh(surface_mesh)
        algorithm.setSubjectPoints(nighresjava.JArray('double')(
                            (surface_mesh['points'].flatten('C')).astype(float)))
        
    if reference_matrix is not None:
        if not isinstance(reference_matrix,np.ndarray): 
            reference_matrix = np.load(reference_matrix)
        
        algorithm.setReferenceMatrix(nighresjava.JArray('double')(
                                (reference_matrix.flatten('C')).astype(float)))
        
        if correspondence_matrix is not None:
            if not isinstance(correspondence_matrix,np.ndarray): 
                correspondence_matrix = np.load(correspondence_matrix)
            
            algorithm.setCorrespondenceMatrix(nighresjava.JArray('double')(
                                (correspondence_matrix.flatten('C')).astype(float)))
            
        if ref_correspondence_matrix is not None:
            if not isinstance(ref_correspondence_matrix,np.ndarray): 
                ref_correspondence_matrix = np.load(ref_correspondence_matrix)
            
            algorithm.setRefCorrespondenceMatrix(nighresjava.JArray('double')(
                                (ref_correspondence_matrix.flatten('C')).astype(float)))
            
        if reference_mesh is not None:
            reference_mesh = load_mesh(reference_mesh)
            algorithm.setReferencePoints(nighresjava.JArray('double')(
                                (reference_mesh['points'].flatten('C')).astype(float)))

    # parameters
    algorithm.setDimensions(dims)
    algorithm.setMatrixSize(msize)
    algorithm.setDistanceScale(scale)
    algorithm.setSpatialScale(space)
    algorithm.setLinkingFactor(link)
    algorithm.setNormalize(normalize)
    
    # execute
    try:
        if reference_matrix is not None:
            if reference_matrix is distance_matrix:
                algorithm.matrixReferenceJointEmbedding()
            else:
                if reference_mesh is not None and surface_mesh is not None:
                    if rotate: 
                        algorithm.matrixRotatedSpatialEmbedding()
                    else:
                        algorithm.matrixSpatialJointEmbedding()
                else:
                    algorithm.matrixSimpleJointEmbedding()
        else:
            algorithm.matrixSimpleEmbedding()
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # Collect output
    npt = distance_matrix.shape[0]
    data = np.reshape(np.array(algorithm.getSubjectEmbeddings(),
                               dtype=np.float64), (npt,dims), 'F')

    if surface_mesh is not None:
        surface_mesh['data'] = data


    if reference_matrix is not None:
        nrf = reference_matrix.shape[0]
        ref_data = np.reshape(np.array(algorithm.getReferenceEmbeddings(),
                                   dtype=np.float64), (nrf,dims), 'F')

        if reference_mesh is not None:
            reference_mesh['data'] = ref_data

    if save_data:
        if (reference_matrix is not None) and (surface_mesh is not None):
            np.save(matrix_file, data)
            np.save(ref_matrix_file, ref_data)
            save_mesh(surf_matrix_file, surface_mesh)
            save_mesh(surf_ref_file, reference_mesh)
            return {'result': matrix_file, 'reference': ref_matrix_file, 'surface': surf_matrix_file, 'ref-surface': surf_ref_file}
        elif surface_mesh is not None:
            np.save(matrix_file, data)
            save_mesh(surf_matrix_file, surface_mesh)
            return {'result': matrix_file, 'surface': surf_matrix_file}
        elif reference_matrix is not None:
            np.save(matrix_file, data)
            np.save(ref_matrix_file, ref_data)
            return {'result': matrix_file, 'reference': ref_matrix_file}
        else:
            np.save(matrix_file, data)
            return {'result': matrix_file}
    else:
        if (reference_matrix is not None) and (surface_mesh is not None):
            return {'result': data, 'reference': ref_data, 'surface': surface_mesh}
        elif surface_mesh is not None:
            return {'result': data, 'surface': surface_mesh}
        elif reference_matrix is not None:
            return {'result': data, 'reference': ref_data}
        else:
            return {'result': data}

