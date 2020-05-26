import os
import sys
import numpy
import nibabel
import nighresjava
from ..io import load_volume, save_volume, load_mesh, save_mesh
from ..utils import _output_dir_4saving, _fname_4saving,_check_available_memory


def parcellation_to_meshes(parcellation_image, connectivity="18/6", 
                     spacing = 0.0, smoothing=1.0,
                     save_data=False, overwrite=False,
                     output_dir=None, file_name=None):

    """Parcellation to meshes

    Creates a collection of triangulated meshes from a parcellation
    using a connectivity-consistent marching cube algorithm.

    Parameters
    ----------
    parcellation_image: niimg
        Parcellation image to be turned into meshes
    connectivity: {"6/18","6/26","18/6","26/6"}, optional
        Choice of digital connectivity to build the mesh (default is 18/6)
    spacing: float, optional
        Added spacing between meshes for better visualization (default is 0.0)
    smoothing: float, optional
        Smoothing of the boundary for prettier meshes, high values may bring 
        small distortions (default is 1.0)
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

        * result ([mesh]): A list of surface mesh dictionaries of "points" and "faces"
          (_p2m-mesh)

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

    print("\nParcellation to Meshes")

    # first we need to know how many meshes to build
    
    # load the data
    p_img = load_volume(parcellation_image)
    p_data = p_img.get_data()
    hdr = p_img.header
    aff = p_img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = p_data.shape

    # count the labels (incl. background)
    labels = numpy.unique(p_data)
    print("found labels: "+str(labels))

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, parcellation_image)

        mesh_files = [] 
        for num,label in enumerate(labels):
            # exclude background as first label
            if num>0:
                mesh_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=parcellation_image,
                                       suffix='p2m-mesh'+str(num),ext="vtk"))
                mesh_files.append(mesh_file)

        if overwrite is False :
            missing = False
            for num,label in enumerate(labels):
                if num>0:
                    if not os.path.isfile(mesh_files[num-1]) :
                        missing = True

            if not missing:
                print("skip computation (use existing results)")
                output = {'result': mesh_files}
                return output

    # start virtual machine if not running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initiate class
    algorithm = nighresjava.SurfaceLevelsetToMesh()

    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])
    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])

    # build a simplified levelset for each structure
    meshes = []
    for num,label in enumerate(labels):
        
        if num>0:
            lvl_data = -1.0*(p_data==label) +1.0*(p_data!=label)
            
            algorithm.setLevelsetImage(nighresjava.JArray('float')(
                                    (lvl_data.flatten('F')).astype(float)))
    
            algorithm.setConnectivity(connectivity)
            algorithm.setZeroLevel(0.0)
            algorithm.setInclusive(True)
            algorithm.setSmoothing(smoothing)

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
            npt = int(numpy.array(algorithm.getPointList(), dtype=numpy.float32).shape[0]/3)
            mesh_points = numpy.reshape(numpy.array(algorithm.getPointList(),
                                   dtype=numpy.float32), (npt,3), 'C')
    
            nfc = int(numpy.array(algorithm.getTriangleList(), dtype=numpy.int32).shape[0]/3)
            mesh_faces = numpy.reshape(numpy.array(algorithm.getTriangleList(),
                                   dtype=numpy.int32), (nfc,3), 'C')
    
            mesh_label = label*numpy.ones((npt,1))
    
            # create the mesh dictionary
            mesh = {"points": mesh_points, "faces": mesh_faces, "data": mesh_label}
    
            meshes.append(mesh)
        
    # if needed, spread values away from center
    if spacing>0:
        center = numpy.zeros((1,3))
        for num,label in enumerate(labels):
            if num>0:
                center = center + numpy.mean(meshes[num-1]['points'], axis=0)
        center = center/(len(labels)-1)
        
        for num,label in enumerate(labels):
            if num>0:
                mesh0 = numpy.mean(meshes[num-1]['points'], axis=0)
                meshes[num-1]['points'] = meshes[num-1]['points']-mesh0 + spacing*(mesh0-center) + center
        
    if save_data:
        for num,label in enumerate(labels):
            if num>0:
                save_mesh(mesh_files[num-1], meshes[num-1])
        return {'result': mesh_files}
    else:
        return {'result': meshes}
