import os
import sys
import numpy as np
import nighresjava
from ..io import load_volume, save_volume, load_mesh_geometry, load_mesh, save_mesh
from ..utils import _output_dir_4saving, _fname_4saving,_check_available_memory


def surface_mesh_mapping(intensity_image, surface_mesh, inflated_mesh=None,
                         mapping_method="closest_point",
                         save_data=False, overwrite=False, output_dir=None,
                         file_name=None):

    """Surface mesh mapping

    Maps volumetric data onto a surface mesh. A second mesh with the same
    graph topology (e.g. an inflated cortical surface) can also be mapped
    with the same data.

    Parameters
    ----------
    intensity_image: niimg
        Intensity image to map onto the surface mesh
    surface_mesh: mesh
        Mesh model of the surface
    inflated_mesh: mesh, optional
        Mesh model of the inflated surface
    mapping_method: {"closest_point","linear_interp","highest_value"}, optional
        Choice of mapping method
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

        * original (mesh): Surface mesh dictionary of "points" and "faces"
          (_map-orig)
        * inflated (mesh): Surface mesh dictionary of "points" and "faces"
          (_map-inf)

    Notes
    ----------
    Ported from original Java module by Pierre-Louis Bazin

    """

    print("\nSurface mesh mapping")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, intensity_image)

        orig_file = os.path.join(output_dir,
                                 _fname_4saving(module=__name__,file_name=file_name,
                                                rootfile=intensity_image,
                                                suffix='map-orig', ext="vtk"))

        inf_file = os.path.join(output_dir,
                                _fname_4saving(module=__name__,file_name=file_name,
                                               rootfile=intensity_image,
                                               suffix='map-inf', ext="vtk"))

        if (overwrite is False and os.path.isfile(orig_file) and
                os.path.isfile(inf_file)):

            print("skip computation (use existing results)")
            output = {'original': orig_file,
                      'inflated': inf_file}
            return output

        elif (overwrite is False and os.path.isfile(orig_file) and
                inflated_mesh is None):

            print("skip computation (use existing results)")
            output = {'original': orig_file,
                      'inflated': None}
            return output

    # start virtual machine if not running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initiate class
    algorithm = nighresjava.CortexSurfaceMeshMapping()

    # load the data
    int_img = load_volume(intensity_image)
    int_data = int_img.get_data()
    hdr = int_img.header
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = int_data.shape

    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])
    if len(dimensions) < 4:
        algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
        nt = 1
    else:
        algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2],
                                dimensions[3])
        nt = dimensions[3]

    algorithm.setIntensityImage(nighresjava.JArray('float')(
                            (int_data.flatten('F')).astype(float)))

    orig_mesh = load_mesh(surface_mesh)

    algorithm.setOriginalSurfacePoints(nighresjava.JArray('float')(
                            (orig_mesh['points'].flatten('C')).astype(float)))
    algorithm.setOriginalSurfaceTriangles(nighresjava.JArray('int')(
                            (orig_mesh['faces'].flatten(
                                'C')).astype(int).tolist()))

    if inflated_mesh is not None:
        inf_mesh = load_mesh(inflated_mesh)

        algorithm.setInflatedSurfacePoints(nighresjava.JArray('float')(
                            (inf_mesh['points'].flatten('C')).astype(float)))
        algorithm.setInflatedSurfaceTriangles(nighresjava.JArray('int')(
                            (inf_mesh['faces'].flatten(
                                'C')).astype(int).tolist()))

    algorithm.setSurfaceConvention("voxels")
    algorithm.setMappingMethod(mapping_method)

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
    npt = int(np.array(algorithm.getMappedOriginalSurfacePoints(),
                       dtype=np.float32).shape[0]/3)
    nfc = int(np.array(algorithm.getMappedOriginalSurfaceTriangles(),
                       dtype=np.int32).shape[0]/3)

    orig_points = np.reshape(np.array(
                            algorithm.getMappedOriginalSurfacePoints(),
                            dtype=np.float32), (npt, 3), 'C')
    orig_faces = np.reshape(np.array(
                            algorithm.getMappedOriginalSurfaceTriangles(),
                            dtype=np.int32), (nfc, 3), 'C')
    orig_data = np.reshape(np.array(
                           algorithm.getMappedOriginalSurfaceValues(),
                           dtype=np.float32), (npt, nt), 'C')

    if inflated_mesh is not None:
        inf_points = np.reshape(np.array(
                                algorithm.getMappedInflatedSurfacePoints(),
                                dtype=np.float32), (npt, 3), 'C')
        inf_faces = np.reshape(np.array(
                               algorithm.getMappedInflatedSurfaceTriangles(),
                               dtype=np.int32), (nfc, 3), 'C')
        inf_data = np.reshape(np.array(
                              algorithm.getMappedInflatedSurfaceValues(),
                              dtype=np.float32), (npt, nt), 'C')

    # create the mesh dictionary
    mapped_orig_mesh = {"points": orig_points, "faces": orig_faces,
                        "data": orig_data}
    if inflated_mesh is not None:
        mapped_inf_mesh = {"points": inf_points, "faces": inf_faces,
                           "data": inf_data}

    if save_data:
        save_mesh(orig_file, mapped_orig_mesh)
        if inflated_mesh is not None:
            save_mesh(inf_file, mapped_inf_mesh)

        if inflated_mesh is not None:
            return {'original': orig_file, 'inflated': inf_file}
        else:
            return {'original': orig_file}
    else:
        if inflated_mesh is not None:
            return {'original': mapped_orig_mesh, 'inflated': mapped_inf_mesh}
        else:
            return {'original': mapped_orig_mesh}
