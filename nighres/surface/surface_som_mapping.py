import os
import sys
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_mesh, save_mesh
from ..utils import _output_dir_4saving, _fname_4saving,_check_available_memory


def surface_som_mapping(surface_mesh, mask_zeros=False,
                            som_size=100, learning_time=100000, total_time=500000,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):

    """Surface SOM mapping

    Maps surface coordinates onto a self-organizing map.

    Parameters
    ----------
    surface_mesh: mesh
        Mesh model of the surface
    mask_zeros: bool
        Whether to mask out zero values (default is False)
    som_size: int
        Size of the 2D SOM to generate
    learning_time: int
        Time for the learning stage iterations
    total_time: int
        Total number of iterations
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

        * original (mesh): Surface mesh dictionary of "points", "faces" and 
          "data" showing the SOM coordinates on the mesh 
        * som (mesh): SOM mesh dictionary of "points", "faces" and "data"
          generated from the SOM grid itself

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin
    
    """

    print("\nSurface som mapping")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, surface_mesh)

        orig_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=surface_mesh,
                                       suffix='som-orig',ext='vtk'))

        som_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=surface_mesh,
                                       suffix='som-grid',ext='vtk'))

        if overwrite is False \
            and os.path.isfile(orig_file) and os.path.isfile(som_file) :
            
            print("skip computation (use existing results)")
            output = {'original': orig_file, 
                      'som': som_file}
            return output
                        
    # start virtual machine if not running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initiate class
    algorithm = nighresjava.SomSurfaceCoordinates()

    # load the data
    orig_mesh = load_mesh(surface_mesh)
    
    algorithm.setSurfacePoints(nighresjava.JArray('float')(
                            (orig_mesh['points'].flatten('C')).astype(float)))
    algorithm.setSurfaceTriangles(nighresjava.JArray('int')(
                            (orig_mesh['faces'].flatten('C')).astype(int).tolist()))
    if orig_mesh['data'] is not None:
        algorithm.setSurfaceValues(nighresjava.JArray('float')(
                            (orig_mesh['data'].flatten('C')).astype(float)))
    
    algorithm.setMaskZeroValues(mask_zeros)
    algorithm.setSomDimension(2)
    algorithm.setSomSize(som_size)
    algorithm.setLearningTime(learning_time)
    algorithm.setTotalTime(total_time)
    
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
    print("collect outputs")
    
    npt = int(np.array(algorithm.getMappedSurfacePoints(), 
                dtype=np.float32).shape[0]/3)
    nfc = int(np.array(algorithm.getMappedSurfaceTriangles(), 
                dtype=np.int32).shape[0]/3) 
    
    print("surface...")
    orig_points = np.reshape(np.array(algorithm.getMappedSurfacePoints(),
                               dtype=np.float32), (npt,3), 'C')
    orig_faces = np.reshape(np.array(algorithm.getMappedSurfaceTriangles(),
                               dtype=np.int32), (nfc,3), 'C')
    orig_data = np.reshape(np.array(algorithm.getMappedSurfaceValues(),
                               dtype=np.float32), (npt,2), 'F')
 
    #    som_points = np.reshape(np.array(algorithm.getMappedSurfacePoints(),
    #                               dtype=np.float32), (npt,3), 'C')
    #    som_faces = np.reshape(np.array(algorithm.getMappedSurfaceTriangles(),
    #                               dtype=np.int32), (nfc,3), 'C')
    #    som_data = np.reshape(np.array(algorithm.getMappedSurfaceValues(),
    #                               dtype=np.float32), (npt,2), 'C')
 
    npt2 = int(np.array(algorithm.getMappedSomPoints(), 
                dtype=np.float32).shape[0]/3)
    nfc2 = int(np.array(algorithm.getMappedSomTriangles(), 
                dtype=np.int32).shape[0]/3)  
    
    print("som... ("+str(npt2)+", "+str(nfc2)+")")
    som_points = np.reshape(np.array(algorithm.getMappedSomPoints(),
                               dtype=np.float32), (npt2,3), 'C')
    som_faces = np.reshape(np.array(algorithm.getMappedSomTriangles(),
                               dtype=np.int32), (nfc2,3), 'C')
    som_data = np.reshape(np.array(algorithm.getMappedSomValues(),
                               dtype=np.float32), (npt2,2), 'F')
    
    # create the mesh dictionary
    mapped_orig_mesh = {"points": orig_points, "faces": orig_faces, 
                        "data": orig_data}
    mapped_som_mesh = {"points": som_points, "faces": som_faces, 
                        "data": som_data}

    if save_data:
        print("saving...")
        save_mesh(orig_file, mapped_orig_mesh)
        save_mesh(som_file, mapped_som_mesh)
        return {'original': orig_file, 'som': som_file}
    else:
        return {'original': mapped_orig_mesh, 'som': mapped_som_mesh}
