import numpy
import nibabel
import os
import sys
import nighresjava
from ..io import load_volume, save_volume, load_mesh, save_mesh
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def mesh_label_inside_distance(surface_mesh, surface_labels=None, selected_labels=None,
                      masked_labels=[0],
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Mesh label distances

    Compute approximate geodesic distances within or across labels defined on a surface mesh

    Parameters
    ----------
    surface_mesh: mesh
        Mesh model of the surface
    surface_labels: array
        Labels associated with the surface, if not included as surface values. 
        Note: distances to labels 0 are not included, and labels -1 are masked
    selected_labels: [int]
        List of labels to use, ignoring the rest (default: use them all)
    masked_labels: [int]
        List of labels to mask (default: mask zero)
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

        * result (mesh): The mesh with distances associated to labels

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    """

    print('\nMesh label inside distance')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, surface_mesh)

        out_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=surface_mesh,
                                   suffix='mli-dist', ext="vtk"))

        if overwrite is False \
            and os.path.isfile(out_file) :
                print("skip computation (use existing results)")
                output = {'result': out_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    algorithm = nighresjava.MeshDistances()

    # set parameters
    
    # load the data
    mesh = load_mesh(surface_mesh)
    
    if surface_labels is not None:
        labels = load_mesh_data(surface_labels)
    else:
        labels = mesh['data']
        
    if masked_labels is not None:
        for lb in masked_labels:
            labels[labels==lb] = -1

    if selected_labels is not None:
        selected = numpy.zeros(labels.shape)
        for lb in selected_labels:
            selected = selected + (labels==lb)*lb
        selected[labels==-1] = -1
        labels = selected

    algorithm.setSurfacePoints(nighresjava.JArray('float')(
                            (mesh['points'].flatten('C')).astype(float)))
    algorithm.setSurfaceTriangles(nighresjava.JArray('int')(
                            (mesh['faces'].flatten('C')).astype(int).tolist()))   
    algorithm.setSurfaceLabels(nighresjava.JArray('int')(
                            (labels.flatten('F')).astype(int).tolist()))
    
       
     # execute the algorithm
    try:
        algorithm.computeInnerDistances()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    
    npt = int(numpy.array(algorithm.getSurfacePoints(), 
                dtype=numpy.float32).shape[0]/3)
    nfc = int(numpy.array(algorithm.getSurfaceTriangles(), 
                dtype=numpy.int32).shape[0]/3) 
    
    points = numpy.reshape(numpy.array(algorithm.getSurfacePoints(),
                               dtype=numpy.float32), (npt,3), 'C')
    faces = numpy.reshape(numpy.array(algorithm.getSurfaceTriangles(),
                               dtype=numpy.int32), (nfc,3), 'C')
    data = numpy.reshape(numpy.array(algorithm.getDistanceValues(),
                               dtype=numpy.float32), (npt), 'F')

    mesh = {"points": points, "faces": faces, 
                        "data": data}
    if save_data:
        save_mesh(out_file, mesh)
        return {'result': out_file}
    else:
        return {'result': mesh}


def mesh_label_outside_distance(surface_mesh, surface_labels=None, selected_labels=None,
                      masked_labels=[0],
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Mesh label distances

    Compute approximate geodesic distances within or across labels defined on a surface mesh

    Parameters
    ----------
    surface_mesh: mesh
        Mesh model of the surface
    surface_labels: array
        Labels associated with the surface, if not included as surface values
        Note: distances to labels 0 are not included, and labels -1 are masked
    selected_labels: [int]
        List of labels to use, ignoring the rest (default: use them all)
    masked_labels: [int]
        List of labels to mask (default: mask zero)
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

        * dist (mesh): The mesh with distances associated to labels
        * label (mesh): The mesh with labels associated to distances

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    """

    print('\nMesh label outside distance')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, surface_mesh)

        dist_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=surface_mesh,
                                   suffix='mlo-dist', ext="vtk"))

        lb_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=surface_mesh,
                                   suffix='mlo-label', ext="vtk"))

        if overwrite is False \
            and os.path.isfile(out_file) :
                print("skip computation (use existing results)")
                output = {'result': out_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    algorithm = nighresjava.MeshDistances()

    # set parameters
    
    # load the data
    mesh = load_mesh(surface_mesh)
    
    if surface_labels is not None:
        labels = load_mesh_data(surface_labels)
    else:
        labels = mesh['data']
        
    if masked_labels is not None:
        for lb in masked_labels:
            labels[labels==lb] = -1

    if selected_labels is not None:
        selected = numpy.zeros(labels.shape)
        for lb in selected_labels:
            selected = selected + (labels==lb)*lb
        selected[labels==-1] = -1
        labels = selected

    algorithm.setSurfacePoints(nighresjava.JArray('float')(
                            (mesh['points'].flatten('C')).astype(float)))
    algorithm.setSurfaceTriangles(nighresjava.JArray('int')(
                            (mesh['faces'].flatten('C')).astype(int).tolist()))   
    algorithm.setSurfaceLabels(nighresjava.JArray('int')(
                            (labels.flatten('F')).astype(int).tolist()))
    
       
     # execute the algorithm
    try:
        algorithm.computeOuterDistances()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    
    npt = int(numpy.array(algorithm.getSurfacePoints(), 
                dtype=numpy.float32).shape[0]/3)
    nfc = int(numpy.array(algorithm.getSurfaceTriangles(), 
                dtype=numpy.int32).shape[0]/3) 
    
    points = numpy.reshape(numpy.array(algorithm.getSurfacePoints(),
                               dtype=numpy.float32), (npt,3), 'C')
    faces = numpy.reshape(numpy.array(algorithm.getSurfaceTriangles(),
                               dtype=numpy.int32), (nfc,3), 'C')
    data = numpy.zeros((npt,3))
    data[:,0] = numpy.reshape(numpy.array(algorithm.getDistanceValuesAt(0),
                               dtype=numpy.float32), (npt), 'F')
    data[:,1] = numpy.reshape(numpy.array(algorithm.getDistanceValuesAt(1),
                               dtype=numpy.float32), (npt), 'F')
    data[:,2] = numpy.reshape(numpy.array(algorithm.getDistanceValuesAt(2),
                               dtype=numpy.float32), (npt), 'F')
    
    label = numpy.zeros((npt,3))
    label[:,0] = numpy.reshape(numpy.array(algorithm.getClosestLabelsAt(0),
                               dtype=numpy.int32), (npt), 'F')
    label[:,1] = numpy.reshape(numpy.array(algorithm.getClosestLabelsAt(1),
                               dtype=numpy.int32), (npt), 'F')
    label[:,2] = numpy.reshape(numpy.array(algorithm.getClosestLabelsAt(2),
                               dtype=numpy.int32), (npt), 'F')

    meshd = {"points": points, "faces": faces, 
                        "data": data}
                        
    meshl = {"points": points, "faces": faces, 
                        "data": label}
    if save_data:
        save_mesh(dist_file, meshd)
        save_mesh(lb_file, meshl)
        return {'dist': dist_file, 'label':lb_file}
    else:
        return {'dist': meshd, 'label': meshl}
