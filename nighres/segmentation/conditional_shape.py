import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def conditional_shape(target_images, structures, contrasts, background=1,
                      shape_atlas_probas=None, shape_atlas_labels=None, 
                      intensity_atlas_hist=None,
                      skeleton_atlas_probas=None, skeleton_atlas_labels=None, 
                      map_to_atlas=None, map_to_target=None,
                      max_iterations=80, max_difference=0.1, ngb_size=4,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Conditioanl Shape Parcellation

    Estimates subcortical structures based on a multi-atlas approach on shape

    Parameters
    ----------
    target_images: [niimg]
        Input images to perform the parcellation from
    structures: int
        Number of structures to parcellate
    contrasts: int
       Number of image intensity contrasts
    background: int
       Number of background tissue classes (default is 1)
    shape_atlas_probas: niimg
        Pre-computed shape atlas from the shape levelsets (replacing them)
    shape_atlas_labels: niimg
        Pre-computed shape atlas from the shape levelsets (replacing them)
    intensity_atlas_hist: niimg
        Pre-computed intensity atlas from the contrast images (replacing them)
    skeleton_atlas_probas: niimg
        Pre-computed skeleton atlas from the shape levelsets (replacing them)
    skeleton_atlas_labels: niimg
        Pre-computed skeleton atlas from the shape levelsets (replacing them)
    map_to_atlas: niimg
        Coordinate mapping from the target to the atlas (opt)
    map_to_target: niimg
        Coordinate mapping from the atlas to the target (opt)
    max_iterations: int
        Maximum number of diffusion iterations to perform
    max_difference: float
        Maximum difference between diffusion steps
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

        * max_spatial_proba (niimg): Maximum spatial probability map (_cspmax-sproba)
        * max_spatial_label (niimg): Maximum spatial probability labels (_cspmax-slabel)
        * max_combined_proba (niimg): Maximum spatial and intensity combined probability map (_cspmax-cproba)
        * max_combined_label (niimg): Maximum spatial and intensity combined probability labels (_cspmax-clabel)
        * max_proba (niimg): Maximum probability map (_cspmax-proba)
        * max_label (niimg): Maximum probability labels (_cspmax-label)
        * neighbors (niimg): Local neighborhood maps (_cspmax-ngb)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    print('\nConditional Shape Parcellation')

    # check topology_lut_dir and set default if not given
    topology_lut_dir = _check_topology_lut_dir(None)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, target_images[0])

        spatial_proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=target_images[0],
                                  suffix='cspmax-sproba', ))

        spatial_label_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=target_images[0],
                                   suffix='cspmax-slabel'))
        
        combined_proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=target_images[0],
                                  suffix='cspmax-cproba', ))

        combined_label_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=target_images[0],
                                   suffix='cspmax-clabel'))
        proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=target_images[0],
                                  suffix='cspmax-proba', ))

        label_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=target_images[0],
                                   suffix='cspmax-label'))

        neighbor_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=target_images[0],
                                   suffix='cspmax-ngb'))
        if overwrite is False \
            and os.path.isfile(spatial_proba_file) \
            and os.path.isfile(spatial_label_file) \
            and os.path.isfile(combined_proba_file) \
            and os.path.isfile(combined_label_file) \
            and os.path.isfile(proba_file) \
            and os.path.isfile(label_file) \
            and os.path.isfile(neighbor_file):
            
            print("skip computation (use existing results)")
            output = {'max_spatial_proba': spatial_proba_file, 
                      'max_spatial_label': spatial_label_file,
                      'max_combined_proba': combined_proba_file, 
                      'max_combined_label': combined_label_file,
                      'max_proba': proba_file, 
                      'max_label': label_file,
                      'neighbors': neighbor_file}
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
    cspmax.setNumberOfSubjectsObjectsBgAndContrasts(1,structures,background,contrasts)
    cspmax.setOptions(True, False, False, False, True)
    cspmax.setDiffusionParameters(max_iterations, max_difference)
    
    # load target image for parameters
    print("load: "+str(target_images[0]))
    img = load_volume(target_images[0])
    data = img.get_data()
    trg_affine = img.get_affine()
    trg_header = img.get_header()
    trg_resolution = [x.item() for x in trg_header.get_zooms()]
    trg_dimensions = data.shape

    cspmax.setTargetDimensions(trg_dimensions[0], trg_dimensions[1], trg_dimensions[2])
    cspmax.setTargetResolutions(trg_resolution[0], trg_resolution[1], trg_resolution[2])

    # target image 1
    cspmax.setTargetImageAt(0, nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))
    
    # if further contrast are specified, input them
    for contrast in range(1,contrasts):    
        print("load: "+str(target_images[contrast]))
        data = load_volume(target_images[contrast]).get_data()
        cspmax.setTargetImageAt(contrast, nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))

    # load the shape and intensity atlases
    print("load: "+str(os.path.join(output_dir,intensity_atlas_hist)))
    hist = load_volume(os.path.join(output_dir,intensity_atlas_hist)).get_data()
    cspmax.setConditionalHistogram(nighresjava.JArray('float')(
                                        (hist.flatten('F')).astype(float)))

    print("load: "+str(os.path.join(output_dir,shape_atlas_probas)))
    
    # load a first image for dim, res
    img = load_volume(os.path.join(output_dir,shape_atlas_probas))
    pdata = img.get_data()
    header = img.get_header()
    affine = img.get_affine()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = pdata.shape
    
    cspmax.setAtlasDimensions(dimensions[0], dimensions[1], dimensions[2])
    cspmax.setAtlasResolutions(resolution[0], resolution[1], resolution[2])

    print("load: "+str(os.path.join(output_dir,shape_atlas_labels)))
    ldata = load_volume(os.path.join(output_dir,shape_atlas_labels)).get_data()
    
    if map_to_target is not None:
        print("map atlas to subject")
        print("load: "+str(map_to_target))
        mdata =  load_volume(map_to_target).get_data()
        cspmax.setMappingToTarget(nighresjava.JArray('float')(
                                            (mdata.flatten('F')).astype(float)))
        
    cspmax.setShapeAtlasProbasAndLabels(nighresjava.JArray('float')(
                                (pdata.flatten('F')).astype(float)),
                                nighresjava.JArray('int')(
                                (ldata.flatten('F')).astype(int).tolist()))

    print("load: "+str(os.path.join(output_dir,skeleton_atlas_probas)))
    pdata = load_volume(os.path.join(output_dir,skeleton_atlas_probas)).get_data()
    
    print("load: "+str(os.path.join(output_dir,skeleton_atlas_labels)))
    ldata = load_volume(os.path.join(output_dir,skeleton_atlas_labels)).get_data()

    cspmax.setSkeletonAtlasProbasAndLabels(nighresjava.JArray('float')(
                                (pdata.flatten('F')).astype(float)),
                                nighresjava.JArray('int')(
                                (ldata.flatten('F')).astype(int).tolist()))

    # execute
    try:
        cspmax.estimateTarget()
        #cspmax.strictSimilarityDiffusion(ngb_size)
        cspmax.fastSimilarityDiffusion(ngb_size)
        #cspmax.fastJointSimilarityDiffusion(ngb_size)
        #cspmax.fastCombinedSimilarityDiffusion(ngb_size)
        #cspmax.globalSmoothing(ngb_size)
        
        cspmax.collapseToJointMaps()
        
        cspmax.precomputeStoppingStatistics(3.0)
        
        cspmax.topologyBoundaryDefinition("wcs", topology_lut_dir)
        
        #cspmax.conditionalVolumeCertaintyThreshold(3.0)
        #cspmax.conditionalPrecomputedVolumeGrowth(3.0)
        cspmax.conditionalPrecomputedDirectVolumeGrowth(3.0)
        
        cspmax.collapseSpatialPriorMaps()
        #cspmax.collapseConditionalMaps()
        #cspmax.collapseToJointMaps()
        
        #cspmax.topologyObjectDefinition("26/6", topology_lut_dir)
        #cspmax.conditionalCollapsedVolumeGrowth(3.0)

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    dimensions = (dimensions[0],dimensions[1],dimensions[2],cspmax.getBestDimension())
    dims3D = (dimensions[0],dimensions[1],dimensions[2])
    dims_ngb = (dimensions[0],dimensions[1],dimensions[2],ngb_size)
    dims3Dtrg = (trg_dimensions[0],trg_dimensions[1],trg_dimensions[2])

    dims3D = dims3Dtrg
    dims_ngb = (trg_dimensions[0],trg_dimensions[1],trg_dimensions[2],ngb_size)
    dims_extra = (trg_dimensions[0],trg_dimensions[1],trg_dimensions[2],4)

    intens_dims = (structures+background,structures+background,contrasts)

    intens_hist_dims = ((structures+background)*(structures+background),cspmax.getNumberOfBins()+6,contrasts)

    spatial_proba_data = np.reshape(np.array(cspmax.getBestSpatialProbabilityMaps(1),
                                   dtype=np.float32), dims3Dtrg, 'F')

    spatial_label_data = np.reshape(np.array(cspmax.getBestSpatialProbabilityLabels(1),
                                    dtype=np.int32), dims3Dtrg, 'F')    

#    combined_proba_data = np.reshape(np.array(cspmax.getBestProbabilityMaps(1),
#                                   dtype=np.float32), dims3Dtrg, 'F')

#    combined_label_data = np.reshape(np.array(cspmax.getBestProbabilityLabels(1),
#                                    dtype=np.int32), dims3Dtrg, 'F')

    combined_proba_data = np.reshape(np.array(cspmax.getJointProbabilityMaps(4),
                                   dtype=np.float32), dims_extra, 'F')

    combined_label_data = np.reshape(np.array(cspmax.getJointProbabilityLabels(4),
                                    dtype=np.int32), dims_extra, 'F')

    proba_data = np.reshape(np.array(cspmax.getFinalProba(),
                                    dtype=np.float32), dims3Dtrg, 'F')

    label_data = np.reshape(np.array(cspmax.getFinalLabel(),
                                    dtype=np.int32), dims3Dtrg, 'F')

    neighbor_data = np.reshape(np.array(cspmax.getNeighborhoodMaps(ngb_size),
                                        dtype=np.float32), dims_ngb, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(spatial_proba_data)
    spatial_proba = nb.Nifti1Image(spatial_proba_data, trg_affine, trg_header)

    header['cal_max'] = np.nanmax(spatial_label_data)
    spatial_label = nb.Nifti1Image(spatial_label_data, trg_affine, trg_header)

    header['cal_max'] = np.nanmax(combined_proba_data)
    combined_proba = nb.Nifti1Image(combined_proba_data, trg_affine, trg_header)

    header['cal_max'] = np.nanmax(combined_label_data)
    combined_label = nb.Nifti1Image(combined_label_data, trg_affine, trg_header)

    trg_header['cal_max'] = np.nanmax(proba_data)
    proba = nb.Nifti1Image(proba_data, trg_affine, trg_header)

    trg_header['cal_max'] = np.nanmax(label_data)
    label = nb.Nifti1Image(label_data, trg_affine, trg_header)

    header['cal_min'] = np.nanmin(neighbor_data)
    header['cal_max'] = np.nanmax(neighbor_data)
    neighbors = nb.Nifti1Image(neighbor_data, trg_affine, trg_header)

    if save_data:
        save_volume(spatial_proba_file, spatial_proba)
        save_volume(spatial_label_file, spatial_label)
        save_volume(combined_proba_file, combined_proba)
        save_volume(combined_label_file, combined_label)
        save_volume(proba_file, proba)
        save_volume(label_file, label)
        save_volume(neighbor_file, neighbors)

        output= {'max_spatial_proba': spatial_proba_file, 'max_spatial_label': spatial_label_file, 
                'max_combined_proba': combined_proba_file, 'max_combined_label': combined_label_file, 
                'max_proba': proba_file, 'max_label': label_file, 'neighbors': neighbor_file}
        return output
    else:
        output= {'max_spatial_proba': spatial_proba, 'max_spatial_label': spatial_label, 
                'max_combined_proba': combined_proba, 'max_combined_label': combined_label, 
                'max_proba': proba, 'max_label': label, 'neighbors': neighbors}
        return output


def conditional_shape_atlasing(subjects, structures, contrasts, 
                      levelset_images=None, skeleton_images=None, 
                      contrast_images=None, background=1,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Conditioanl Shape Parcellation Atlasing

    Builds a multi-atlas prior for conditional shape parcellation

    Parameters
    ----------
    subjects: int
        Number of atlas subjects
    structures: int
        Number of structures to parcellate
    contrasts: int
       Number of image intensity contrasts
    levelset_images: [niimg]
        Atlas shape levelsets indexed by (subjects,structures)
    skeleton_images: [niimg]
        Atlas shape skeletons indexed by (subjects,structures)
    contrast_images: [niimg]
        Atlas images to use in the parcellation, indexed by (subjects, contrasts)
    background: int
        Number of separate tissue classes for the background (default is 1)
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

        * max_spatial_proba (niimg): Maximum spatial probability map (_cspmax-sproba)
        * max_spatial_label (niimg): Maximum spatial probability labels (_cspmax-slabel)
        * cond_hist (niimg): Conditional intensity histograms (_cspmax-chist)
        * max_skeleton_proba (niimg): Maximum skeleton probability map (_cspmax-kproba)
        * max_skeleton_label (niimg): Maximum skeleton probability labels (_cspmax-klabel)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    print('\nConditional Shape Atlasing')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, contrast_images[0][0])

        spatial_proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=contrast_images[0][0],
                                  suffix='cspmax-sproba', ))

        spatial_label_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=contrast_images[0][0],
                                   suffix='cspmax-slabel'))

        condhist_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=contrast_images[0][0],
                                   suffix='cspmax-chist'))
        
        skeleton_proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=contrast_images[0][0],
                                  suffix='cspmax-kproba', ))

        skeleton_label_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=contrast_images[0][0],
                                   suffix='cspmax-klabel'))

        
        if overwrite is False \
            and os.path.isfile(spatial_proba_file) \
            and os.path.isfile(spatial_label_file) \
            and os.path.isfile(condhist_file) \
            and os.path.isfile(skeleton_proba_file) \
            and os.path.isfile(skeleton_label_file):
            
            print("skip computation (use existing results)")
            output = {'max_spatial_proba': spatial_proba_file, 
                      'max_spatial_label': spatial_label_file,
                      'cond_hist': condhist_file,
                      'max_skeleton_proba': skeleton_proba_file, 
                      'max_skeleton_label': skeleton_label_file}

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
    cspmax.setNumberOfSubjectsObjectsBgAndContrasts(subjects,structures,background,contrasts)
    cspmax.setOptions(True, False, False, False, True)
     
    # load target image for parameters
    # load a first image for dim, res
    img = load_volume(contrast_images[0][0])
    data = img.get_data()
    header = img.get_header()
    affine = img.get_affine()
    trg_resolution = [x.item() for x in header.get_zooms()]
    trg_dimensions = data.shape
    
    cspmax.setTargetDimensions(trg_dimensions[0], trg_dimensions[1], trg_dimensions[2])
    cspmax.setTargetResolutions(trg_resolution[0], trg_resolution[1], trg_resolution[2])

    resolution = trg_resolution
    dimensions = trg_dimensions
        
    cspmax.setAtlasDimensions(dimensions[0], dimensions[1], dimensions[2])
    cspmax.setAtlasResolutions(resolution[0], resolution[1], resolution[2])
    
    # load the atlas structures and contrasts, if needed
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
    # execute first step
    scale = 1.0
    try:
        scale = cspmax.computeAtlasPriors()
 
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # clean up and go to second step
    levelset_images = None
    contrast_images = None
    
    for sub in range(subjects):
        for struct in range(structures):
            print("load: "+str(skeleton_images[sub][struct]))
            data = load_volume(skeleton_images[sub][struct]).get_data()
            cspmax.setSkeletonImageAt(sub, struct, nighresjava.JArray('float')(
                                                (data.flatten('F')).astype(float)))
                
    try:
        cspmax.computeSkeletonPriors(scale)
 
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    skeleton_images = None

    # reshape output to what nibabel likes
    dimensions = (dimensions[0],dimensions[1],dimensions[2],cspmax.getBestDimension())
    dimskel = (dimensions[0],dimensions[1],dimensions[2],int(cspmax.getBestDimension()/4))
    dims3Dtrg = (trg_dimensions[0],trg_dimensions[1],trg_dimensions[2])

    intens_dims = (structures+background,structures+background,contrasts)
    intens_hist_dims = ((structures+background)*(structures+background),cspmax.getNumberOfBins()+6,contrasts)

    spatial_proba_data = np.reshape(np.array(cspmax.getBestSpatialProbabilityMaps(dimensions[3]),
                                   dtype=np.float32), dimensions, 'F')

    spatial_label_data = np.reshape(np.array(cspmax.getBestSpatialProbabilityLabels(dimensions[3]),
                                    dtype=np.int32), dimensions, 'F')    

    intens_hist_data = np.reshape(np.array(cspmax.getConditionalHistogram(),
                                       dtype=np.float32), intens_hist_dims, 'F')

    skeleton_proba_data = np.reshape(np.array(cspmax.getBestSkeletonProbabilityMaps(dimskel[3]),
                                   dtype=np.float32), dimskel, 'F')

    skeleton_label_data = np.reshape(np.array(cspmax.getBestSkeletonProbabilityLabels(dimskel[3]),
                                    dtype=np.int32), dimskel, 'F')    


    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(spatial_proba_data)
    spatial_proba = nb.Nifti1Image(spatial_proba_data, affine, header)

    header['cal_max'] = np.nanmax(spatial_label_data)
    spatial_label = nb.Nifti1Image(spatial_label_data, affine, header)

    chist = nb.Nifti1Image(intens_hist_data, None, None)

    header['cal_max'] = np.nanmax(skeleton_proba_data)
    skeleton_proba = nb.Nifti1Image(skeleton_proba_data, affine, header)

    header['cal_max'] = np.nanmax(skeleton_label_data)
    skeleton_label = nb.Nifti1Image(skeleton_label_data, affine, header)

    if save_data:
        save_volume(spatial_proba_file, spatial_proba)
        save_volume(spatial_label_file, spatial_label)
        save_volume(condhist_file, chist)
        save_volume(skeleton_proba_file, skeleton_proba)
        save_volume(skeleton_label_file, skeleton_label)
        output= {'max_spatial_proba': spatial_proba_file, 
                 'max_spatial_label': spatial_label_file, 
                 'cond_hist': condhist_file,
                 'max_skeleton_proba': skeleton_proba_file, 
                 'max_skeleton_label': skeleton_label_file}
        return output
    else:
        output= {'max_spatial_proba': spatial_proba, 
                 'max_spatial_label': spatial_label, 
                 'cond_hist': chist,
                 'max_skeleton_proba': skeleton_proba, 
                 'max_skeleton_label': skeleton_label}
        return output

def conditional_shape_updating(subjects, structures, contrasts, 
                      levelset_images=None, skeleton_images=None, 
                      contrast_images=None, 
                      atlas_weight=10.0, update_weight=10.0, 
                      shape_atlas_probas=None, shape_atlas_labels=None, 
                      intensity_atlas_hist=None,
                      skeleton_atlas_probas=None, skeleton_atlas_labels=None, 
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Conditioanl Shape Parcellation Atlas Updating

    Updates a multi-atlas prior for conditional shape parcellation

    Parameters
    ----------
    subjects: int
        Number of atlas subjects
    structures: int
        Number of structures to parcellate
    contrasts: int
       Number of image intensity contrasts
    levelset_images: [niimg]
        Atlas shape levelsets indexed by (subjects,structures)
    skeleton_images: [niimg]
        Atlas shape skeletons indexed by (subjects,structures)
    contrast_images: [niimg]
        Atlas images to use in the parcellation, indexed by (subjects, contrasts)
    atlas_weight: float
        Weighting factor for the current atlas (e.g. number of subjects already included)
    update_weight: float
        Weighting factor for the new update (e.g. number of subjects to be added)
    shape_atlas_probas: niimg
        Pre-computed shape atlas from the shape levelsets (replacing them)
    shape_atlas_labels: niimg
        Pre-computed shape atlas from the shape levelsets (replacing them)
    intensity_atlas_hist: niimg
        Pre-computed intensity atlas from the contrast images (replacing them)
    skeleton_atlas_probas: niimg
        Pre-computed skeleton atlas from the shape levelsets (replacing them)
    skeleton_atlas_labels: niimg
        Pre-computed skeleton atlas from the shape levelsets (replacing them)
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

        * max_spatial_proba (niimg): Maximum spatial probability map (_cspmax-sproba)
        * max_spatial_label (niimg): Maximum spatial probability labels (_cspmax-slabel)
        * cond_hist (niimg): Conditional intensity histograms (_cspmax-chist)
        * max_skeleton_proba (niimg): Maximum skeleton probability map (_cspmax-kproba)
        * max_skeleton_label (niimg): Maximum skeleton probability labels (_cspmax-klabel)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    print('\nConditional Shape Updating')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, contrast_images[0][0])

        spatial_proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=contrast_images[0][0],
                                  suffix='cspmax-sproba', ))

        spatial_label_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=contrast_images[0][0],
                                   suffix='cspmax-slabel'))

        condhist_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=contrast_images[0][0],
                                   suffix='cspmax-chist'))
        
        skeleton_proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=contrast_images[0][0],
                                  suffix='cspmax-kproba', ))

        skeleton_label_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=contrast_images[0][0],
                                   suffix='cspmax-klabel'))

        
        if overwrite is False \
            and os.path.isfile(spatial_proba_file) \
            and os.path.isfile(spatial_label_file) \
            and os.path.isfile(condhist_file) \
            and os.path.isfile(skeleton_proba_file) \
            and os.path.isfile(skeleton_label_file):
            
            print("skip computation (use existing results)")
            output = {'max_spatial_proba': spatial_proba_file, 
                      'max_spatial_label': spatial_label_file,
                      'cond_hist': condhist_file,
                      'max_skeleton_proba': skeleton_proba_file, 
                      'max_skeleton_label': skeleton_label_file}

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
    cspmax.setNumberOfSubjectsObjectsBgAndContrasts(subjects,structures,3,contrasts)
    cspmax.setOptions(True, False, False, False, True)
     
    # load target image for parameters
    # load a first image for dim, res
    img = load_volume(contrast_images[0][0])
    data = img.get_data()
    header = img.get_header()
    affine = img.get_affine()
    trg_resolution = [x.item() for x in header.get_zooms()]
    trg_dimensions = data.shape
    
    cspmax.setTargetDimensions(trg_dimensions[0], trg_dimensions[1], trg_dimensions[2])
    cspmax.setTargetResolutions(trg_resolution[0], trg_resolution[1], trg_resolution[2])

    resolution = trg_resolution
    dimensions = trg_dimensions
        
    cspmax.setAtlasDimensions(dimensions[0], dimensions[1], dimensions[2])
    cspmax.setAtlasResolutions(resolution[0], resolution[1], resolution[2])
    
    # load the shape and intensity atlases
    print("load: "+str(os.path.join(output_dir,intensity_atlas_hist)))
    hist = load_volume(os.path.join(output_dir,intensity_atlas_hist)).get_data()
    cspmax.setConditionalHistogram(nighresjava.JArray('float')(
                                        (hist.flatten('F')).astype(float)))

    print("load: "+str(os.path.join(output_dir,shape_atlas_probas)))
    pdata = load_volume(os.path.join(output_dir,shape_atlas_probas)).get_data()
    print("load: "+str(os.path.join(output_dir,shape_atlas_labels)))
    ldata = load_volume(os.path.join(output_dir,shape_atlas_labels)).get_data()
    
    cspmax.setShapeAtlasProbasAndLabels(nighresjava.JArray('float')(
                                (pdata.flatten('F')).astype(float)),
                                nighresjava.JArray('int')(
                                (ldata.flatten('F')).astype(int).tolist()))

    print("load: "+str(os.path.join(output_dir,skeleton_atlas_probas)))
    pdata = load_volume(os.path.join(output_dir,skeleton_atlas_probas)).get_data()
    
    print("load: "+str(os.path.join(output_dir,skeleton_atlas_labels)))
    ldata = load_volume(os.path.join(output_dir,skeleton_atlas_labels)).get_data()

    cspmax.setSkeletonAtlasProbasAndLabels(nighresjava.JArray('float')(
                                (pdata.flatten('F')).astype(float)),
                                nighresjava.JArray('int')(
                                (ldata.flatten('F')).astype(int).tolist()))

    # load the atlas structures and contrasts, if needed
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
    # execute first step
    scale = 1.0
    try:
        scale = cspmax.updateAtlasPriors(atlas_weight, update_weight)
 
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # clean up and go to second step
    levelset_images = None
    contrast_images = None
    
    for sub in range(subjects):
        for struct in range(structures):
            print("load: "+str(skeleton_images[sub][struct]))
            data = load_volume(skeleton_images[sub][struct]).get_data()
            cspmax.setSkeletonImageAt(sub, struct, nighresjava.JArray('float')(
                                                (data.flatten('F')).astype(float)))
                
    try:
        cspmax.updateSkeletonPriors(scale, atlas_weight, update_weight)
 
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    skeleton_images = None

    # reshape output to what nibabel likes
    dimensions = (dimensions[0],dimensions[1],dimensions[2],cspmax.getBestDimension())
    dimskel = (dimensions[0],dimensions[1],dimensions[2],int(cspmax.getBestDimension()/4))
    dims3Dtrg = (trg_dimensions[0],trg_dimensions[1],trg_dimensions[2])

    intens_dims = (structures+1,structures+1,contrasts)
    intens_hist_dims = ((structures+1)*(structures+1),cspmax.getNumberOfBins()+6,contrasts)

    spatial_proba_data = np.reshape(np.array(cspmax.getBestSpatialProbabilityMaps(dimensions[3]),
                                   dtype=np.float32), dimensions, 'F')

    spatial_label_data = np.reshape(np.array(cspmax.getBestSpatialProbabilityLabels(dimensions[3]),
                                    dtype=np.int32), dimensions, 'F')    

    intens_hist_data = np.reshape(np.array(cspmax.getConditionalHistogram(),
                                       dtype=np.float32), intens_hist_dims, 'F')

    skeleton_proba_data = np.reshape(np.array(cspmax.getBestSkeletonProbabilityMaps(dimskel[3]),
                                   dtype=np.float32), dimskel, 'F')

    skeleton_label_data = np.reshape(np.array(cspmax.getBestSkeletonProbabilityLabels(dimskel[3]),
                                    dtype=np.int32), dimskel, 'F')    


    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(spatial_proba_data)
    spatial_proba = nb.Nifti1Image(spatial_proba_data, affine, header)

    header['cal_max'] = np.nanmax(spatial_label_data)
    spatial_label = nb.Nifti1Image(spatial_label_data, affine, header)

    chist = nb.Nifti1Image(intens_hist_data, None, None)

    header['cal_max'] = np.nanmax(skeleton_proba_data)
    skeleton_proba = nb.Nifti1Image(skeleton_proba_data, affine, header)

    header['cal_max'] = np.nanmax(skeleton_label_data)
    skeleton_label = nb.Nifti1Image(skeleton_label_data, affine, header)

    if save_data:
        save_volume(spatial_proba_file, spatial_proba)
        save_volume(spatial_label_file, spatial_label)
        save_volume(condhist_file, chist)
        save_volume(skeleton_proba_file, skeleton_proba)
        save_volume(skeleton_label_file, skeleton_label)
        output= {'max_spatial_proba': spatial_proba_file, 
                 'max_spatial_label': spatial_label_file, 
                 'cond_hist': condhist_file,
                 'max_skeleton_proba': skeleton_proba_file, 
                 'max_skeleton_label': skeleton_label_file}
        return output
    else:
        output= {'max_spatial_proba': spatial_proba, 
                 'max_spatial_label': spatial_label, 
                 'cond_hist': chist,
                 'max_skeleton_proba': skeleton_proba, 
                 'max_skeleton_label': skeleton_label}
        return output
