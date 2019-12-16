import os
import sys
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_volume, save_volume, load_mesh_geometry, save_mesh_geometry
from ..utils import _output_dir_4saving, _fname_4saving,_check_available_memory


def mesh_to_levelset(surface_mesh, reference_image, 
                     save_data=False, overwrite=False,
                     output_dir=None, file_name=None):

    """Mesh to levelset

    Creates a signed distance function from a triangulated mesh using pseudonormals.

    Parameters
    ----------
    surface_mesh: mesh
        Mesh model of the surface
    reference_image: niimg
        Image of the dimensions and resolutions corresponding to the mesh
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

        * result (niimg): Levelset function representing the mesh (_m2l-lvl)

    Notes
    ----------
    Ported from original Java module by Christine Tardif and Pierre-Louis Bazin. 

    References
    ----------
    """

    print("\nMesh to Levelset")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, surface_mesh)

        lvl_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=surface_mesh,
                                       suffix='m2l-lvl'))

        if overwrite is False \
            and os.path.isfile(lvl_file) :

            print("skip computation (use existing results)")
            output = {'result': lvl_file}
            return output

    # start virtual machine if not running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initiate class
    algorithm = nighresjava.SurfaceMeshToLevelsetPseudoNormals()

    # load the data
    mesh = load_mesh_geometry(surface_mesh)
    
    algorithm.setSurfacePoints(nighresjava.JArray('float')(
                            (mesh['points'].flatten('C')).astype(float)))
    algorithm.setSurfaceTriangles(nighresjava.JArray('int')(
                            (mesh['faces'].flatten('C')).astype(int).tolist()))
    
    ref_img = load_volume(reference_image)
    hdr = ref_img.header
    aff = ref_img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = hdr.get_data_shape()

    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])
    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])

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
    lvl_data = np.reshape(np.array(algorithm.getLevelsetImage(),
                                    dtype=np.float32), dimensions, 'F')

    # create the mesh dictionary
    header['cal_min'] = np.nanmin(lvl_data)
    header['cal_max'] = np.nanmax(lvl_data)
    lvl = nb.Nifti1Image(lvl_data, affine, header)

    if save_data:
        save_volume(lvl_file, lvl)
        return {'result': lvl_file}
    else:
        return {'result': lvl}
        