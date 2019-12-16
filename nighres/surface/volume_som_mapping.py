import os
import sys
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_mesh, save_mesh, load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving,_check_available_memory


def volume_som_mapping(proba_image,
                            som_size=100, learning_time=100000, total_time=500000,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):

    """Surface SOM mapping

    Maps surface coordinates onto a self-organizing map.

    Parameters
    ----------
    proba_image: nii
        Probabilistic representation of the surface
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

        * map (nii): Volumetric image showing the SOM coordinates map
        * som (mesh): SOM mesh dictionary of "points", "faces" and "data"
          generated from the SOM grid itself

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin
    
    """

    print("\nVolume som mapping")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, proba_image)

        map_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=proba_image,
                                       suffix='som-orig'))

        som_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=proba_image,
                                       suffix='som-grid',ext='vtk'))

        if overwrite is False \
            and os.path.isfile(map_file) and os.path.isfile(som_file) :
            
            print("skip computation (use existing results)")
            output = {'map': map_file, 
                      'som': som_file}
            return output
                        
    # start virtual machine if not running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initiate class
    algorithm = nighresjava.SomVolumeCoordinates()

    # load the data
    prob_img = load_volume(proba_image)
    prob_data = prob_img.get_data()
    hdr = prob_img.header
    aff = prob_img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = prob_data.shape
    
    algorithm.setProbaImage(nighresjava.JArray('float')(
                                    (prob_data.flatten('F')).astype(float)))
    
    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])
    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    
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

    print("volume...")
    dimensions = (dimensions[0],dimensions[1],dimensions[2],2)
    map_data = np.reshape(np.array(algorithm.getMappedImage(),
                               dtype=np.float32), dimensions, 'F')

    hdr['cal_max'] = np.nanmax(map_data)
    mapped_img = nb.Nifti1Image(map_data, aff, hdr)

    npt = int(np.array(algorithm.getMappedSomPoints(), 
                dtype=np.float32).shape[0]/3)
    nfc = int(np.array(algorithm.getMappedSomTriangles(), 
                dtype=np.int32).shape[0]/3)  
    
    print("som... ("+str(npt)+", "+str(nfc)+")")
    som_points = np.reshape(np.array(algorithm.getMappedSomPoints(),
                               dtype=np.float32), (npt,3), 'C')
    som_faces = np.reshape(np.array(algorithm.getMappedSomTriangles(),
                               dtype=np.int32), (nfc,3), 'C')
    som_data = np.reshape(np.array(algorithm.getMappedSomValues(),
                               dtype=np.float32), (npt,2), 'F')
    
    # create the mesh dictionary
    mapped_som_mesh = {"points": som_points, "faces": som_faces, 
                        "data": som_data}

    if save_data:
        print("saving...")
        save_volume(map_file, mapped_img)
        save_mesh(som_file, mapped_som_mesh)

        return {'map': map_file, 'som': som_file}
    else:
        return {'map': mapped_img, 'som': mapped_som_mesh}
