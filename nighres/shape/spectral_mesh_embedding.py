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


def spectral_mesh_embedding(surface_mesh, 
                    reference_mesh=None,
                    dims=3,
                    msize=200,
                    scale=100.0,
                    save_data=False, 
                    overwrite=False, 
                    output_dir=None,
                    file_name=None):

    """ Spectral mesh embedding
    
    Derive a spectral Laplacian embedding from a surface mesh, 
    optionally taking a reference mesh into account.

    Parameters
    ----------
    surface_mesh: mesh
        Mesh model of the surface
    reference_mesh: mesh
        Mesh model of the reference
    dims: int
        Number of kept dimensions in the representation (default is 1)
    msize: int
        Target matrix size for subsampling (default is 800)
    scale: float
        Distance scale between sample points (default is 100.0)
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

        * result (mesh): Coordinate map (_sme-coord)

    Notes
    ----------
    
    References
    ----------

    """

    print("\nSpectral Mesh Embedding")

    if save_data:
        output_dir = _output_dir_4saving(output_dir, surface_mesh)

        mesh_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=surface_mesh,
                                  suffix='sme-coord',ext='vtk'))

        if reference_mesh is not None:
            ref_mesh_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=surface_mesh,
                                  suffix='sme-ref',ext='vtk'))

        if overwrite is False \
            and os.path.isfile(mesh_file) :
                print("skip computation (use existing results)")
                output = {'result': mesh_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    algorithm = nighresjava.SpectralMeshDenseJointEmbedding()

    # load the data
    mesh = load_mesh(surface_mesh)
    
    algorithm.setSurfacePoints(nighresjava.JArray('float')(
                            (mesh['points'].flatten('C')).astype(float)))
    algorithm.setSurfaceTriangles(nighresjava.JArray('int')(
                            (mesh['faces'].flatten('C')).astype(int).tolist()))

    if reference_mesh is not None:
        
        ref_mesh = load_mesh(reference_mesh)
        
        algorithm.setReferencePoints(nighresjava.JArray('float')(
                                (ref_mesh['points'].flatten('C')).astype(float)))
        algorithm.setReferenceTriangles(nighresjava.JArray('int')(
                                (ref_mesh['faces'].flatten('C')).astype(int).tolist()))
    
    algorithm.setDimensions(dims)
    
    algorithm.setMatrixSize(msize)
    
    algorithm.setDistanceScale(scale)
    
    # execute
    try:
        if reference_mesh is not None:
            #algorithm.pointDistanceJointEmbedding()
            algorithm.pointDistanceJointRotatedEmbedding()
        else:
            #algorithm.pointDistanceEmbedding()
            algorithm.pointDistanceSparseEmbedding()
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # Collect output
    npt = mesh['points'].shape[0]
    mesh_points = mesh['points']
    mesh_faces = mesh['faces']    
    mesh_data = np.reshape(np.array(algorithm.getEmbeddingValues(),
                               dtype=np.float32), (npt,dims), 'F')
    # create the mesh dictionary
    mesh = {"points": mesh_points, "faces": mesh_faces, "data": mesh_data}

    if reference_mesh is not None:
        nrf = ref_mesh['points'].shape[0]
        ref_mesh_points = ref_mesh['points']
        ref_mesh_faces = ref_mesh['faces']    
        ref_mesh_data = np.reshape(np.array(algorithm.getReferenceEmbeddingValues(),
                                   dtype=np.float32), (nrf,dims), 'F')
        # create the mesh dictionary
        ref_mesh = {"points": ref_mesh_points, "faces": ref_mesh_faces, "data": ref_mesh_data}
        

    if save_data:
        if reference_mesh is not None:
            save_mesh(mesh_file, mesh)
            save_mesh(ref_mesh_file, ref_mesh)
            return {'result': mesh_file, 'reference': ref_mesh_file}
        else:
            save_mesh(mesh_file, mesh)
            return {'result': mesh_file}
    else:
        if reference_mesh is not None:
            return {'result': mesh, 'reference': ref_mesh}
        else:
            return {'result': mesh}



def spectral_mesh_spatial_embedding(surface_mesh, 
                    reference_mesh=None,
                    dims=3,
                    msize=2000,
                    scale=10.0,
                    link=1.0,
                    depth=12,
                    alpha=0.0,
                    eigengame=True,
                    save_data=False, 
                    overwrite=False, 
                    output_dir=None,
                    file_name=None):

    """ Spectral mesh spatial embedding
    
    Derive a spectral Laplacian embedding from a surface mesh.

    Parameters
    ----------
    surface_mesh: mesh
        Mesh model of the surface
    reference_mesh: mesh
        Mesh model of the reference (optional)
    dims: int
        Number of kept dimensions in the representation (default is 1)
    msize: int
        Target matrix size for subsampling (default is 2000)
    scale: float
        Distance scale between sample points (default is 10.0)
    link: float
        Spatial linking factor (default is 1.0)
    depth: int
        Number of nearest neighbors used in first approximation (default is 12)
    alpha: float
        Laplacian norm parameter in [0:1] (default is 0.0)
    eigengame: bool, optional
        Run the eigengame full scale decomposition (default is True)
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

        * result (mesh): Coordinate map (_sme-coord)

    Notes
    ----------
    
    References
    ----------

    """

    print("\nSpectral Mesh Spatial Embedding")

    if save_data:
        output_dir = _output_dir_4saving(output_dir, surface_mesh)

        mesh_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=surface_mesh,
                                  suffix='sme-coord',ext='vtk'))
        
        if reference_mesh is not None:
            ref_mesh_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=surface_mesh,
                                  suffix='sme-ref',ext='vtk'))

        if overwrite is False \
            and os.path.isfile(mesh_file) :
                print("skip computation (use existing results)")
                output = {'result': mesh_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    algorithm = nighresjava.SpectralMeshEmbedding()

    # load the data
    mesh = load_mesh(surface_mesh)
    
    algorithm.setSurfacePoints(nighresjava.JArray('float')(
                            (mesh['points'].flatten('C')).astype(float)))
    algorithm.setSurfaceTriangles(nighresjava.JArray('int')(
                            (mesh['faces'].flatten('C')).astype(int).tolist()))

    if reference_mesh is not None:
        
        ref_mesh = load_mesh(reference_mesh)
        
        algorithm.setReferencePoints(nighresjava.JArray('float')(
                                (ref_mesh['points'].flatten('C')).astype(float)))
        algorithm.setReferenceTriangles(nighresjava.JArray('int')(
                                (ref_mesh['faces'].flatten('C')).astype(int).tolist()))
    
    algorithm.setDimensions(dims)
    
    algorithm.setMatrixSize(msize)
    
    algorithm.setDistanceScale(scale)
    
    algorithm.setLinkingFactor(link)
    
    # execute
    try:
        if reference_mesh is not None:
            #algorithm.meshDistanceSparseEmbedding2(depth,eigengame,False,alpha);
            #algorithm.meshDistanceReferenceSparseEmbedding(depth,eigengame,False,alpha);
            algorithm.meshDistanceJointSparseEmbedding(depth,eigengame,False,alpha);
        else:
            algorithm.meshDistanceSparseEmbedding2(depth,eigengame,False,alpha);
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # Collect output
    npt = mesh['points'].shape[0]
    mesh_points = mesh['points']
    mesh_faces = mesh['faces']    
    mesh_data = np.reshape(np.array(algorithm.getEmbeddingValues(),
                               dtype=np.float32), (npt,dims), 'F')
    # create the mesh dictionary
    mesh = {"points": mesh_points, "faces": mesh_faces, "data": mesh_data}

    if reference_mesh is not None:
        nrf = ref_mesh['points'].shape[0]
        ref_mesh_points = ref_mesh['points']
        ref_mesh_faces = ref_mesh['faces']    
        ref_mesh_data = np.reshape(np.array(algorithm.getReferenceEmbeddingValues(),
                                   dtype=np.float32), (nrf,dims), 'F')
        # create the mesh dictionary
        ref_mesh = {"points": ref_mesh_points, "faces": ref_mesh_faces, "data": ref_mesh_data}
        

    if save_data:
        if reference_mesh is not None:
            save_mesh(mesh_file, mesh)
            save_mesh(ref_mesh_file, ref_mesh)
            return {'result': mesh_file, 'reference': ref_mesh_file}
        else:
            save_mesh(mesh_file, mesh)
            return {'result': mesh_file}
    else:
        if reference_mesh is not None:
            return {'result': mesh, 'reference': ref_mesh}
        else:
            return {'result': mesh}


