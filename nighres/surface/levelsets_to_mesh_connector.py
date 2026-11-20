import os
import sys
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_volume, save_volume, load_mesh_geometry, save_mesh_geometry
from ..utils import _output_dir_4saving, _fname_4saving,_check_available_memory


def levelsets_to_mesh_connector(levelset_image1, levelset_image2, label='lvl1_lvl2', length=6.0, side=0.5, distance=1.0,
                     save_data=False, overwrite=False,
                     output_dir=None, file_name=None):

    """Levelsets to mesh connector

    Creates a triangulated mesh of a polugonal connector between tow levelset surfaces.

    Parameters
    ----------
    levelset_image1: niimg
        First levelset image to be connected
    levelset_image2: niimg
        Second levelset image to be connected
    label: str, optional
        Name to give to the connector
    length: float, optional
       Length of the connector to be specified (default is 6.0)
    side: float, optional
       Radius of the connector to be specified (default is 0.5)
    distance: float, optional
        Distance to the levelsets to define the interface (default is 1.0)
    save_data: bool, optional
        Save output data to file (default is False)
    overwrite: bool, optional
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

        * result (mesh): Surface mesh dictionary of "points" and "faces"
          (_l2c-mesh)

    Notes
    ----------
    Ported from original Java module by Pierre-Louis Bazin.

    References
    """

    print("\nLevelsets to Mesh Connector")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, levelset_image1)

        mesh_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=label,
                                       suffix='l2c-mesh',ext="obj"))

        if overwrite is False \
            and os.path.isfile(mesh_file) :

            print("skip computation (use existing results)")
            output = {'result': mesh_file}
            return output

    # start virtual machine if not running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initiate class
    algorithm = nighresjava.LevelsetsToMeshConnector()

    # load the data
    lvl1_img = load_volume(levelset_image1)
    lvl1_data = lvl1_img.get_fdata()
    hdr = lvl1_img.header
    aff = lvl1_img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = lvl1_data.shape

    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])
    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])

    algorithm.setLevelsetImage1(nighresjava.JArray('float')(
                            (lvl1_data.flatten('F')).astype(float)))

    lvl2_data = load_volume(levelset_image2).get_fdata()
    algorithm.setLevelsetImage2(nighresjava.JArray('float')(
                            (lvl2_data.flatten('F')).astype(float)))

    algorithm.setConnectorLength(length)
    algorithm.setConnectorSide(side)
    algorithm.setBoundaryDistance(distance)

    # execute class
    try:
        algorithm.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # collect outputs, if any
    if algorithm.isBoundaryFound():
        npt = int(np.array(algorithm.getPointList(), dtype=np.float32).shape[0]/3)
        mesh_points = np.reshape(np.array(algorithm.getPointList(),
                                   dtype=np.float32), shape=(npt,3), order='C')
    
        nfc = int(np.array(algorithm.getTriangleList(), dtype=np.int32).shape[0]/3)
        mesh_faces = np.reshape(np.array(algorithm.getTriangleList(),
                                   dtype=np.int32), shape=(nfc,3), order='C')
    
        # create the mesh dictionary
        mesh = {"points": mesh_points, "faces": mesh_faces}
    
        if save_data:
            save_mesh_geometry(mesh_file, mesh)
            return {'result': mesh_file}
        else:
            return {'result': mesh}
    else:
        return {'result': None}
       