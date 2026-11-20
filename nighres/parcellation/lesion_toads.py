import numpy
import nibabel
import os
import sys
import json
import math
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory
from nighres.global_settings import DEFAULT_LESIONTOADS_ATLAS
from nighres.data.download_data import download_LesionTOADS_atlas

 
def lesion_toads(target_images, modalities,
                  atlas_file=None, smoothing=0.2, 
                  max_iterations=80, max_difference=0.0001, 
                  max_gm_dist=3, max_interventricle_dist=25, 
                  max_ventricle_dist=2, atlas_prior=2.0,
                  align="multi_fully_affine", connectivity="18/6",
                  save_data=False, overwrite=False, output_dir=None,
                  file_name=None):
        
    """ Lesion TOpology-preserving Anatomy-Driven Segmentation (LesionTOADS)

    Estimates brain structures and white matter hyper-intensities using a topology-preserving
    brain model

    Parameters
    ----------
    target_images: [niimg]
        Input images to perform the parcellation from
    modalities: [str]
        Modalitiy of each of the input images, taken from the ones defined in the atlas:
        {"T1_MPRAGE", "T1_SPGR", "FLAIR"} 
    atlas_file: str
        Location of the LesionTOADS atlas to use (default: None to use the default version)
    results: str
        Output maps computed, taken from: "hard segmentation","hard segmentation+memberships",
        "cruise inputs","dura removal inputs" (default is "hard segmentation").
    align: str
        Type of atlas alignment from {"rigid","multi_fully_affine"} (default "multi_fully_affine")
    max_iterations: int
        Maximum number of diffusion iterations to perform
    max_difference: float
        Maximum difference between diffusion steps
    connectivity: str
        Topological connectivity from {"18/6", "6/18", "26/6", "6/26"} (default is "18/6")
    intensity_prior: float
        Importance scaling factor for the intensities in [0,1] (default is 1.0)
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

        * classification (niimg): Maximum classification labels (_ltoads-class)
        * lesions (niimg): Lesion mask (_ltoads-lesion)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    # allowing only one type of output for now
    result_type="hard segmentation"
    #result_type="hard segmentation+memberships"
    outputClassificationFromMembership = True
    includeLesionnClassification = True
    
    # (numerous) parameters that should stay fixed
    outliers=0.1
    doGrowing = True
    lowEratio = math.sqrt(max_difference)
    bgthreshold = 0.0
    
    firstLimit = 3.0
    lastLimit = 0.5
    spread = 1.0
    atlas_range = 0.2
    
    register = True
    levels = 4
    init_align_iterations = 50
    main_align_iterations = 1
    rel_mode = "diffeomorphism"
    demon_scale = 2.0
    demon_smoothing = 1.0
    norm_type = "Identity"
    centroid_mode = "prior"
    centroid_smoothness = 0.1
    correct=True
    output_field=False
    correction_method = "Chebyshev"
    polynomial_degree = 3
    kernel_size = 30.0
    distance_mode = "normal distance"
    #distance_mode = "no distance"
    #distance_mode = "narrow_band_competition"
    use_lesion_weight=True
		

    print('\nLesionTOADS')

    # check topology_lut_dir and set default if not given
    topology_lut_dir = _check_topology_lut_dir(None)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, target_images[0])

        class_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=target_images[0],
                                  suffix='ltoads-class', ))

        mems_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=target_images[0],
                                  suffix='ltoads-mems', ))

        lesion_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=target_images[0],
                                  suffix='ltoads-lesion', ))

        if overwrite is False \
            and os.path.isfile(class_file) and os.path.isfile(mems_file) and os.path.isfile(lesion_file):
            
            print("skip computation (use existing results)")
            output = {'classification': class_file, 'maxmem': mems_file, 'lesions': lesion_file}
            return output

    contrasts = len(target_images)

    # if not specified, check if standard atlases are available or download them
    if ( (atlas_file is None)):
        
        if (not (os.path.exists(DEFAULT_LESIONTOADS_ATLAS))):
            download_LesionTOADS_atlas(overwrite=False)
            
        atlas_file = DEFAULT_LESIONTOADS_ATLAS

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance & set parameters
    ltoads = nighresjava.LesionToads(contrasts, modalities, atlas_file, result_type,
                                    smoothing, outliers, max_iterations, max_difference,
                                    bgthreshold, outputClassificationFromMembership,
                                    firstLimit, lastLimit, spread,
                                    max_gm_dist, max_interventricle_dist,  max_ventricle_dist,
                                    includeLesionnClassification, atlas_prior, atlas_range,
                                    rel_mode, centroid_mode, centroid_smoothness, 
                                    use_lesion_weight, align, norm_type, register, levels,
                                    init_align_iterations, main_align_iterations, demon_smoothing,
                                    demon_scale, correct, output_field, correction_method,
                                    polynomial_degree, kernel_size, distance_mode, connectivity, topology_lut_dir)
    
    # load target image for parameters
    print("load: "+str(target_images[0]))
    img = load_volume(target_images[0])
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    orient = nibabel.aff2axcodes(affine)
    ori_code = [0,0,0]
    for idx in range(3):
        if orient[idx]=='R': ori_code[idx]=1
        elif orient[idx]=='L': ori_code[idx]=2
        elif orient[idx]=='P': ori_code[idx]=3
        elif orient[idx]=='A': ori_code[idx]=4
        elif orient[idx]=='S': ori_code[idx]=5
        elif orient[idx]=='I': ori_code[idx]=6
    
    axis = 0
    if ori_code[2]==5 or ori_code[2]==6: axis = 10
    elif ori_code[2]==3 or ori_code[2]==4: axis = 20
    elif ori_code[2]==1 or ori_code[2]==2: axis = 30

    ltoads.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    ltoads.setResolutions(resolution[0], resolution[1], resolution[2])
    ltoads.setOrientations(ori_code[0], ori_code[1], ori_code[2], axis)

    # target image 1
    ltoads.setSourceImageAt(0, nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    
    # if further contrast are specified, input them
    for contrast in range(1,contrasts):    
        print("load: "+str(target_images[contrast]))
        data = load_volume(target_images[contrast]).get_fdata()
        ltoads.setSourceImageAt(contrast, nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))


    # execute
    try:
        ltoads.calcSegmentation()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    class_data = numpy.reshape(numpy.array(ltoads.getResultImageAt(0),
                                    dtype=numpy.int32), shape=dimensions, order='F')

    #dims4d = (dimensions[0],dimensions[1],dimensions[2],14)
    #mems_data = numpy.reshape(numpy.array(ltoads.getResultImageAt(0),
    #                                dtype=numpy.float32), shape=dims4d, order='F')
    mems_data = numpy.reshape(numpy.array(ltoads.getResultImageAt(1),
                                    dtype=numpy.int32), shape=dimensions, order='F')

    lesion_data = numpy.reshape(numpy.array(ltoads.getResultImageAt(2),
                                    dtype=numpy.int32), shape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = numpy.nanmax(class_data)
    class_img = nibabel.Nifti1Image(class_data, affine, header)

    header['cal_max'] = numpy.nanmax(mems_data)
    mems_img = nibabel.Nifti1Image(mems_data, affine, header)

    header['cal_max'] = numpy.nanmax(lesion_data)
    lesion_img = nibabel.Nifti1Image(lesion_data, affine, header)

    if save_data:
        save_volume(class_file, class_img)
        save_volume(mems_file, mems_img)
        save_volume(lesion_file, lesion_img)

        output= {'classification': class_file, 'maxmem': mems_file, 'lesions': lesion_file}
        return output
    else:
        output= {'classification': class_img, 'maxmem': mems_img, 'lesions': lesion_img}
        return output

