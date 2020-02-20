import os
import sys
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_volume, save_volume, load_mesh_geometry, save_mesh_geometry
from ..utils import _output_dir_4saving, _fname_4saving,_check_available_memory


def levelset_to_mesh(levelset_image, connectivity="18/6", level=0.0,
                     inclusive=True, save_data=False, overwrite=False,
                     output_dir=None, file_name=None):

    """Levelset to mesh

    Creates a triangulated mesh from the distance to a levelset surface
    representation using a connectivity-consistent marching cube algorithm.

    Parameters
    ----------
    levelset_image: niimg
        Levelset image to be turned into a mesh
    connectivity: {"6/18","6/26","18/6","26/6"}, optional
        Choice of digital connectivity to build the mesh (default is 18/6)
    level: float, optional
        Value of the levelset function to use as isosurface (default is 0)
    inclusive: bool, optional
        Whether voxels at the exact 'level' value are inside the isosurface
        (default is True)
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
          (_l2m-mesh)

    Notes
    ----------
    Ported from original Java module by Pierre-Louis Bazin. Original algorithm
    from [1]_ and adapted from [2]_.

    References
    ----------
    .. [1] Han et al (2003). A Topology Preserving Level Set Method for
        Geometric Deformable Models
        doi:
    .. [2] Lucas et al (2010). The Java Image Science Toolkit (JIST) for
        Rapid Prototyping and Publishing of Neuroimaging Software
        doi:
    """

    print("\nLevelset to Mesh")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, levelset_image)

        mesh_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=levelset_image,
                                       suffix='l2m-mesh',ext="vtk"))

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
    algorithm = nighresjava.SurfaceLevelsetToMesh()

    # load the data
    lvl_img = load_volume(levelset_image)
    lvl_data = lvl_img.get_data()
    hdr = lvl_img.header
    aff = lvl_img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = lvl_data.shape

    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])
    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])

    algorithm.setLevelsetImage(nighresjava.JArray('float')(
                            (lvl_data.flatten('F')).astype(float)))

    algorithm.setConnectivity(connectivity)
    algorithm.setZeroLevel(level)
    algorithm.setInclusive(inclusive)

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
    npt = int(np.array(algorithm.getPointList(), dtype=np.float32).shape[0]/3)
    mesh_points = np.reshape(np.array(algorithm.getPointList(),
                               dtype=np.float32), (npt,3), 'C')

    nfc = int(np.array(algorithm.getTriangleList(), dtype=np.int32).shape[0]/3)
    mesh_faces = np.reshape(np.array(algorithm.getTriangleList(),
                               dtype=np.int32), (nfc,3), 'C')

    # create the mesh dictionary
    mesh = {"points": mesh_points, "faces": mesh_faces}

    if save_data:
        save_mesh_geometry(mesh_file, mesh)
        return {'result': mesh_file}
    else:
        return {'result': mesh}
