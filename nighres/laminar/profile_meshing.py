import os
import sys
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_volume, save_volume, load_mesh_geometry, save_mesh, save_mesh_geometry
from ..utils import _output_dir_4saving, _fname_4saving,_check_available_memory


def profile_meshing(profile_surface_image, starting_surface_mesh, 
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):

    """Profile meshing

    Creates a point-by-point matched set of 3D meshes, one for each of layer of
    the layer boundary surface, with the mesh topology of the starting mesh.
    The starting mesh should be inside the cortex, for instance the mid-layer
    surface from volumetric layering.

    Parameters
    ----------
    profile_surface_image: niimg
        4D image containing levelset representations of different intracortical
        surfaces on which data should be sampled
    starting_surface_mesh: mesh
        Mesh model of the surface
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

        * profile ([mesh]): Collection of surface mesh dictionary of "points" and "faces"
          (_mesh-p#)

    Notes
    ----------
    Ported from original Java module by Pierre-Louis Bazin

    """

    print("\nProfile meshing")

    # check number of layers
    nlayers = load_volume(profile_surface_image).header.get_data_shape()[3]

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, profile_surface_image)

        mesh_files = []
        for n in range(nlayers):
            mesh_files.append(os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=profile_surface_image,
                                       suffix='mesh-p'+str(n),ext="vtk")))

        if overwrite is False :
            missing = False
            for n in range(nlayers):
                if not os.path.isfile(mesh_files[n]):
                    missing = True

            if not missing:
                print("skip computation (use existing results)")
                output = {'profile': mesh_files}
                return output

    # start virtual machine if not running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initiate class
    algorithm = nighresjava.LaminarProfileMeshing()

    # load the data
    surface_img = load_volume(profile_surface_image)
    surface_data = surface_img.get_data()
    hdr = surface_img.header
    aff = surface_img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = surface_data.shape

    algorithm.setProfileSurfaceImage(nighresjava.JArray('float')(
                                   (surface_data.flatten('F')).astype(float)))
    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])
    algorithm.setDimensions(dimensions[0], dimensions[1],
                          dimensions[2], dimensions[3])

    orig_mesh = load_mesh_geometry(starting_surface_mesh)

    algorithm.setInputSurfacePoints(nighresjava.JArray('float')(
                            (orig_mesh['points'].flatten('C')).astype(float)))
    algorithm.setInputSurfaceTriangles(nighresjava.JArray('int')(
                            (orig_mesh['faces'].flatten('C')).astype(int).tolist()))

    algorithm.setSurfaceConvention("voxels")

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
    npt = int(orig_mesh['points'].shape[0])
    nfc = int(orig_mesh['faces'].shape[0])

    meshes = []
    lines = np.zeros((nlayers,npt,3))
    for n in range(nlayers):
        points = np.reshape(np.array(algorithm.getSampledSurfacePoints(n),
                               dtype=np.float32), (npt,3), 'C')
        faces = np.reshape(np.array(algorithm.getSampledSurfaceTriangles(n),
                               dtype=np.int32), (nfc,3), 'C')
        # create the mesh dictionary
        meshes.append({"points": points, "faces": faces})

        lines[n,:,:] = points
        
        if save_data:
            save_mesh_geometry(mesh_files[n], meshes[n])
 
    if save_data:
        _write_profiles_vtk("mesh_lines.vtk",lines)
 
    if save_data:
        return {'profile': mesh_files}
    else:
        return {'profile': meshes}

def _write_profiles_vtk(filename, vertices, decimation=10):
    '''
    Creates ASCII coded vtk file from numpy arrays using pandas.
    Inputs:
    -------
    (mandatory)
    * filename: str, path to location where vtk file should be stored
    * vertices: numpy array with profile vertex coordinates,  shape (n_profiles, n_vertices, 3)
    Usage:
    ---------------------
    _write_vtk('/path/to/vtk/file.vtk', v_array)
    '''

    vertices = vertices[:,0:-1:decimation,:]

    import pandas as pd
    # infer number of vertices and faces
    number_profiles = vertices.shape[0]
    number_vertices = vertices.shape[1]
    # make header and subheader dataframe
    header = ['# vtk DataFile Version 3.0',
              'None',
              'ASCII',
              'DATASET POLYDATA',
              'POINTS %i float' % (number_vertices*number_vertices)
              ]
    header_df = pd.DataFrame(header)
    sub_header = ['LINES %i %i' % (number_vertices, (number_profiles+1) * number_vertices)]
    sub_header_df = pd.DataFrame(sub_header)
    # make dataframe from vertices
    vertex_df = pd.DataFrame(np.reshape(vertices, (number_profiles*number_vertices,3)))
    # make dataframe from faces, appending first row of 3's
    # (indicating the polygons are triangles)
    lines = np.reshape(number_profiles * (np.ones(number_vertices)), (number_vertices, 1))
    lines = lines.astype(int)
    print("lines: "+str(lines.shape))
    indices = np.zeros((number_vertices, number_profiles))
    for p in range(number_profiles):
        indices[:,p] = range(p*number_vertices,p*number_vertices+number_vertices)
    print("indices: "+str(indices.shape))
    lines_df = pd.DataFrame(np.concatenate((lines, indices), axis=1))
    print("lines: "+str(lines_df.shape))
    # write dfs to csv
    header_df.to_csv(filename, header=None, index=False)
    with open(filename, 'a') as f:
        vertex_df.to_csv(f, header=False, index=False, float_format='%.3f',
                         sep=' ')
    with open(filename, 'a') as f:
        sub_header_df.to_csv(f, header=False, index=False)
    with open(filename, 'a') as f:
        lines_df.to_csv(f, header=False, index=False, float_format='%.0f',
                        sep=' ')
