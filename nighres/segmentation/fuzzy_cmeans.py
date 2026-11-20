import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def fuzzy_cmeans(image, clusters=3, max_iterations=50, max_difference=0.01, 
                    smoothing=0.1, fuzziness=2.0, mask_zero=True, map_intensity=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Fuzzy C-means image segmentation

    Estimates intensity clusters with spatial smoothness and partial voluming.
    Based on the RFCM algorithm of (Pham, 2001) [1]_.

    Parameters
    ----------
    image: niimg
        Input image to segment
    clusters: int
        Number of clusters to estimate (default is 3)
    max_iterations: int
        Maximum number of iterations to perform (default is 50)
    max_difference: float
        Maximum difference between steps for stopping (default is 0.01)
    smoothing: float
        Ratio of spatial smoothness to impose on the clusters (default is 0.1)
    fuzziness: float
        Scaling of the C-means measure, in [1.0 - 3.0] (default is 2.0)
    mask_zero: bool
        Whether to ignore zero values (default is True)
    map_intensity: bool
        Whether to compute a centroid-based intensity image (default is False)
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

        * memberships [niimg]: List of membership functions for each cluster in [0,1] (_rfcm-mem#cluster)
        * classification (niimg): Hard classification of most likely cluster per voxel (_rfcm-class)
        * intensity (niimg): Centroid-based intensity map, if created (_rfcm-map)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    References
    ----------
    .. [1] D.L. Pham, Spatial Models for Fuzzy Clustering,
       CVIU, vol. 84, pp. 285--297, 2001
    """

    print('\nFuzzy C-means')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        mem_files = []
        for c in range(clusters):
            mem_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=image,
                                  suffix='rfcm-mem'+str(c+1), ))
            mem_files.append(mem_file)

        classification_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='rfcm-class'))
        if map_intensity:
            intensity_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=image,
                                       suffix='rfcm-map'))
            
        if overwrite is False \
            and os.path.isfile(classification_file):
            
            missing = False
            for mem_file in mem_files:
                if not os.path.isfile(mem_file):
                    missing = True
            if map_intensity and not os.path.isfile(intensity_file):     
                missing = True
            if not missing:        
                print("skip computation (use existing results)")
                if map_intensity:
                    output = {'classification': classification_file, 
                          'memberships': mem_files,
                          'intensity': intensity_file}
                else:
                    output = {'classification': classification_file, 
                          'memberships': mem_files}
                return output


    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    rfcm = nighresjava.FuzzyCmeans()

    # set parameters
    rfcm.setClusterNumber(clusters)
    rfcm.setSmoothing(smoothing)
    rfcm.setFuzziness(fuzziness)
    rfcm.setMaxDist(max_difference)
    rfcm.setMaxIter(max_iterations)
    
    # load target image for parameters
    print("load: "+str(image))
    img = load_volume(image)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    if len(dimensions)>2: rfcm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    else: rfcm.setDimensions(dimensions[0], dimensions[1], 1)
    
    if len(resolution)>2: rfcm.setResolutions(resolution[0], resolution[1], resolution[2])
    else: rfcm.setResolutions(resolution[0], resolution[1], resolution[1])

    # image
    rfcm.setImage(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    
    # execute
    try:
        if mask_zero: rfcm.initZeroMaskImage()
        else: rfcm.initBasicMaskImage()
        rfcm.execute()
        if map_intensity:
            rfcm.interpolateCentroidIntensity()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    classification_data = np.reshape(np.array(rfcm.getClassification(),
                                   dtype=np.int32), shape=dimensions, order='F')

    #header['cal_max'] = np.nanmax(classification_data)
    classification = nb.Nifti1Image(classification_data, affine, header)

    memberships = []
    for c in range(clusters):
        mem_data = np.reshape(np.array(rfcm.getMembership(c),
                                    dtype=np.float32), shape=dimensions, order='F')    
        #header['cal_max'] = np.nanmax(mem_data)
        membership = nb.Nifti1Image(mem_data, affine, header)
        memberships.append(membership)

    if map_intensity:
        intens_data = np.reshape(np.array(rfcm.getClassImage(),
                                   dtype=np.float32), shape=dimensions, order='F')

        #header['cal_min'] = np.nanmin(intens_data)
        #header['cal_max'] = np.nanmax(intens_data)
        intensity = nb.Nifti1Image(intens_data, affine, header)

    if save_data:
        save_volume(classification_file, classification)
        for c in range(clusters):
            save_volume(mem_files[c], memberships[c])
        if map_intensity:
            save_volume(intensity_file,intensity)
            output= {'classification': classification_file, 'memberships': mem_files,
                     'intensity': intensity_file}
        else:
            output= {'classification': classification_file, 'memberships': mem_files}
    else:
        if map_intensity:
            output= {'classification': classification, 'memberships': memberships,
                     'intensity': intensity}
        else:
            output= {'classification': classification, 'memberships': memberships}

    return output

def fuzzy_cmeans_multi(images, clusters=3, max_iterations=50, max_difference=0.01, 
                    smoothing=0.1, fuzziness=2.0, mask_zero=True, map_intensity=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Fuzzy C-means image segmentation

    Estimates intensity clusters with spatial smoothness and partial voluming.
    Based on the RFCM algorithm of (Pham, 2001) [1]_.

    Parameters
    ----------
    images: [niimg]
        Input images to segment (note: the first one sets the classification ordering)
    clusters: int
        Number of clusters to estimate (default is 3)
    max_iterations: int
        Maximum number of iterations to perform (default is 50)
    max_difference: float
        Maximum difference between steps for stopping (default is 0.01)
    smoothing: float
        Ratio of spatial smoothness to impose on the clusters (default is 0.1)
    fuzziness: float
        Scaling of the C-means measure, in [1.0 - 3.0] (default is 2.0)
    mask_zero: bool
        Whether to ignore zero values (default is True)
    map_intensity: bool
        Whether to compute a centroid-based intensity image (default is False)
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

        * memberships [niimg]: List of membership functions for each cluster in [0,1] (_rfcm-mem#cluster)
        * classification (niimg): Hard classification of most likely cluster per voxel (_rfcm-class)
        * intensity [niimg]: Centroid-based intensity maps, if created (_rfcm-map)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    References
    ----------
    .. [1] D.L. Pham, Spatial Models for Fuzzy Clustering,
       CVIU, vol. 84, pp. 285--297, 2001
    """

    print('\nFuzzy C-means')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, images[0])

        mem_files = []
        for c in range(clusters):
            mem_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=images[0],
                                  suffix='rfcm-mem'+str(c+1), ))
            mem_files.append(mem_file)

        classification_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=images[0],
                                   suffix='rfcm-class'))
        int_files = []
        if map_intensity:
            for img in images:
                intensity_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=img,
                                       suffix='rfcm-map'))
                int_files.append(intensity_file)
            
        if overwrite is False \
            and os.path.isfile(classification_file):
            
            missing = False
            for mem_file in mem_files:
                if not os.path.isfile(mem_file):
                    missing = True
            if map_intensity:
                for int_file in int_files:
                    if not os.path.isfile(int_file):
                        missing = True
            if not missing:        
                print("skip computation (use existing results)")
                if map_intensity:
                    output = {'classification': classification_file, 
                          'memberships': mem_files,
                          'intensity': int_files}
                else:
                    output = {'classification': classification_file, 
                          'memberships': mem_files}
                return output


    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    rfcm = nighresjava.FuzzyCmeansMulti()

    # set parameters
    rfcm.setClusterNumber(clusters)
    rfcm.setSmoothing(smoothing)
    rfcm.setFuzziness(fuzziness)
    rfcm.setMaxDist(max_difference)
    rfcm.setMaxIter(max_iterations)
    
    rfcm.setImageNumber(len(images))

    # load target image for parameters
    print("load: "+str(images[0]))
    img = load_volume(images[0])
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    if len(dimensions)>2: rfcm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    else: rfcm.setDimensions(dimensions[0], dimensions[1], 1)
    
    if len(resolution)>2: rfcm.setResolutions(resolution[0], resolution[1], resolution[2])
    else: rfcm.setResolutions(resolution[0], resolution[1], resolution[1])

    # images
    for idx,img in enumerate(images):
        data = load_volume(img).get_fdata()
        rfcm.setImageAt(idx,nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    
    # execute
    try:
        if mask_zero: rfcm.initZeroMaskImage()
        else: rfcm.initBasicMaskImage()
        rfcm.execute()
        if map_intensity:
            rfcm.interpolateCentroidIntensities()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    classification_data = np.reshape(np.array(rfcm.getClassification(),
                                   dtype=np.int32), shape=dimensions, order='F')

    #header['cal_max'] = np.nanmax(classification_data)
    classification = nb.Nifti1Image(classification_data, affine, header)

    memberships = []
    for c in range(clusters):
        mem_data = np.reshape(np.array(rfcm.getMembership(c),
                                    dtype=np.float32), shape=dimensions, order='F')    
        #header['cal_max'] = np.nanmax(mem_data)
        membership = nb.Nifti1Image(mem_data, affine, header)
        memberships.append(membership)

    intensities = []
    if map_intensity:
        for i in range(len(images)):
            intens_data = np.reshape(np.array(rfcm.getClassImageAt(i),
                                   dtype=np.float32), shape=dimensions, order='F')

            #header['cal_min'] = np.nanmin(intens_data)
            #header['cal_max'] = np.nanmax(intens_data)
            intensity = nb.Nifti1Image(intens_data, affine, header)
            intensities.append(intensity)

    if save_data:
        save_volume(classification_file, classification)
        for c in range(clusters):
            save_volume(mem_files[c], memberships[c])
        if map_intensity:
            for i in range(len(images)):
                save_volume(int_files[i],intensities[i])
            output= {'classification': classification_file, 'memberships': mem_files,
                     'intensity': int_files}
        else:
            output= {'classification': classification_file, 'memberships': mem_files}
    else:
        if map_intensity:
            output= {'classification': classification, 'memberships': memberships,
                     'intensity': intensities}
        else:
            output= {'classification': classification, 'memberships': memberships}

    return output
