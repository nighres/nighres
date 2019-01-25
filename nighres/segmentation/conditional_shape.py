import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def conditional_shape(target_images, levelset_images, contrast_images, 
                      subjects, structures, contrasts,
                      cancel_bg=False, cancel_all=False, 
                      sum_proba=False, max_proba=False,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Conditioanl Shape Parcellation

    Estimates subcortical structures based on a multi-atlas approach on shape

    Parameters
    ----------
    target_images: [niimg]
        Input images to perform the parcellation from
    levelset_images: [niimg]
        Atlas shape levelsets indexed by (subjects,structures)
    contrast_images: [niimg]
        Atlas images to use in the parcellation, indexed by (subjects, contrasts)
    subjects: int
        Number of atlas subjects
    structures: int
        Number of structures to parcellate
    contrasts: int
       Number of image intensity contrasts
    cancel_bg: bool
        Cancel the main background class (default is False)
    cancel_all: bool
        Cancel all main classes (default is False)
    sum_proba: bool
        Output the sum of conditional probabilities (default is False)
    max_proba: bool
        Output the max of conditional probabilities (default is False)
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

        * max_proba (niimg): Maximum probability map (_cspmax-proba)
        * max_label (niimg): Maximum probability labels (_cspmax-label)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    print('\nConditional Shape Parcellation')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, target_images[0])

        proba_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                  rootfile=target_images[0],
                                  suffix='cspmax-proba', ))

        label_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=target_images[0],
                                   suffix='cspmax-label'))
        if overwrite is False \
            and os.path.isfile(proba_file) \
            and os.path.isfile(label_file) :
            
            print("skip computation (use existing results)")
            output = {'max_proba': load_volume(proba_file), 
                      'max_label': load_volume(label_file)}
            return output


    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    cspmax = nighresjava.ConditionalShapeSegmentation()

    # set parameters
    cspmax.setNumberOfSubjectsObjectsAndContrasts(subjects,structures,contrasts)
    cspmax.setOptions(True, cancel_bg, cancel_all, sum_proba, max_proba)
    
    # load target image for parameters
    #print("load: "+str(target_images[0]))
    img = load_volume(target_images[0])
    data = img.get_data()
    affine = img.get_affine()
    header = img.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    cspmax.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    cspmax.setResolutions(resolution[0], resolution[1], resolution[2])

    # target image 1
    cspmax.setTargetImageAt(0, nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    
    # if further contrast are specified, input them
    for contrast in range(1,contrasts):    
        print("load: "+str(target_images[contrast]))
        data = load_volume(target_images[contrast]).get_data()
        cspmax.setTargetImageAt(contrast, nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))

    # load the atlas structures and contrasts
    for sub in range(subjects):
        for struct in range(structures):
            print("load: "+str(levelset_images[sub][struct]))
            data = load_volume(levelset_images[sub][struct]).get_data()
            cspmax.setLevelsetImageAt(sub, struct, nighresjava.JArray('float')(
                                                (data.flatten('F')).astype(float)))
                
        for contrast in range(contrasts):
            print("load: "+str(contrast_images[sub][contrast]))
            data = load_volume(contrast_images[sub][contrast]).get_data()
            cspmax.setContrastImageAt(sub, contrast, nighresjava.JArray('float')(
                                                (data.flatten('F')).astype(float)))

    dimensions = (dimensions[0],dimensions[1],dimensions[2],cspmax.getBestDimension())

    # execute
    try:
        cspmax.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    proba_data = np.reshape(np.array(cspmax.getBestProbabilityMaps(),
                                   dtype=np.float32), dimensions, 'F')

    label_data = np.reshape(np.array(cspmax.getBestProbabilityLabels(),
                                    dtype=np.int32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(proba_data)
    proba = nb.Nifti1Image(proba_data, affine, header)

    header['cal_max'] = np.nanmax(label_data)
    label = nb.Nifti1Image(label_data, affine, header)

    if save_data:
        save_volume(proba_file, proba)
        save_volume(label_file, label)

    return {'max_proba': proba, 'max_label': label}
