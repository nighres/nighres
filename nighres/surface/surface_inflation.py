import os
import sys
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_mesh, save_mesh
from ..utils import _output_dir_4saving, _fname_4saving,_check_available_memory


def surface_inflation(surface_mesh, step_size=0.75, max_iter=2000, max_curv=10.0,
                        method='area', regularization=0.0,
                        save_data=False, overwrite=False, output_dir=None,
                        file_name=None):

    """Surface inflation

    Inflate a surface with the method of Tosun et al _[1].

    Parameters
    ----------
    surface_mesh: mesh
        Mesh model of the surface
    step_size: float
        Relaxation rate in [0, 1]: values closer to 1 are more stable but slower 
        (default is 0.75)
    max_iter: int
        Maximum number of iterations (default is 2000)
    max_curv: float
        Desired maximum curvature (default is 10.0)   
    method: str
        Method used for averaging: 'area' based on the area of the triangle,
        'dist' based on the vertex distance, 'numv' based on number of vertices
        (default is 'area')
    regularization: float
        Regularization parameter for reducing local singularities (default is 0.0)
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

        * result (mesh): Surface mesh dictionary of "points", "faces" and 
          "data" showing the SOM coordinates on the mesh 

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin
    
    """

    print("\nSurface inflation")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, surface_mesh)

        infl_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=surface_mesh,
                                       suffix='infl-mesh',ext='vtk'))

        if overwrite is False \
            and os.path.isfile(infl_file) :
            
            print("skip computation (use existing results)")
            output = {'result': infl_file}
            return output
                        
    # start virtual machine if not running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initiate class
    algorithm = nighresjava.SurfaceInflation()

    # load the data
    orig_mesh = load_mesh(surface_mesh)
    
    algorithm.setSurfacePoints(nighresjava.JArray('float')(
                            (orig_mesh['points'].flatten('C')).astype(float)))
    algorithm.setSurfaceTriangles(nighresjava.JArray('int')(
                            (orig_mesh['faces'].flatten('C')).astype(int).tolist()))
    
    algorithm.setStepSize(step_size)
    algorithm.setMaxIter(max_iter)
    algorithm.setMaxCurv(max_curv)
    if method=='area':
        algorithm.setWeightingMethod(algorithm.AREA)
    elif method=='dist':
        algorithm.setWeightingMethod(algorithm.DIST)
    else:
        algorithm.setWeightingMethod(algorithm.NUMV)
        
    algorithm.setRegularization(regularization)
        
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
    
    npt = int(np.array(algorithm.getInflatedSurfacePoints(), 
                dtype=np.float32).shape[0]/3)
    nfc = int(np.array(algorithm.getInflatedSurfaceTriangles(), 
                dtype=np.int32).shape[0]/3) 
    
    print("surface...")
    orig_points = np.reshape(np.array(algorithm.getInflatedSurfacePoints(),
                               dtype=np.float32), (npt,3), 'C')
    orig_faces = np.reshape(np.array(algorithm.getInflatedSurfaceTriangles(),
                               dtype=np.int32), (nfc,3), 'C')
    orig_data = np.reshape(np.array(algorithm.getInflatedSurfaceValues(),
                               dtype=np.float32), (npt), 'F')
 
     
    # create the mesh dictionary
    inflated_orig_mesh = {"points": orig_points, "faces": orig_faces, 
                        "data": orig_data}

    if save_data:
        print("saving...")
        save_mesh(infl_file, inflated_orig_mesh)
        return {'result': infl_file}
    else:
        return {'result': inflated_orig_mesh}
