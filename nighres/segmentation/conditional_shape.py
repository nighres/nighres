import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def conditional_shape(target_images, structures, contrasts,
                      shape_atlas_probas=None, shape_atlas_labels=None, 
                      intensity_atlas_hist=None,
                      atlas_space=False, adjust_volume=False,
                      map_to_atlas=None, map_to_target=None,
                      max_iterations=20, max_difference=0.01, ngb_size=4,
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
    shape_atlas_probas: niimg
        Pre-computed shape atlas from the shape levelsets (replacing them)
    shape_atlas_labels: niimg
        Pre-computed shape atlas from the shape levelsets (replacing them)
    intensity_atlas_hist: niimg
        Pre-computed intensity atlas from the contrast images (replacing them)
    atlas_space: bool
        Whether to estimate the labeling in atlas space (default is False)
    adjust_volume: bool
        Whether to estimate the final segmentation based on volume optimization
        (default is False)
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

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, target_images[0])

        spatial_proba_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                  rootfile=target_images[0],
                                  suffix='cspmax-sproba', ))

        spatial_label_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=target_images[0],
                                   suffix='cspmax-slabel'))
        combined_proba_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                  rootfile=target_images[0],
                                  suffix='cspmax-cproba', ))

        combined_label_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=target_images[0],
                                   suffix='cspmax-clabel'))
        proba_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                  rootfile=target_images[0],
                                  suffix='cspmax-proba', ))

        label_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=target_images[0],
                                   suffix='cspmax-label'))

        neighbor_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
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
            output = {'max_spatial_proba': load_volume(spatial_proba_file), 
                      'max_spatial_label': load_volume(spatial_label_file),
                      'max_combined_proba': load_volume(combined_proba_file), 
                      'max_combined_label': load_volume(combined_label_file),
                      'max_proba': load_volume(proba_file), 
                      'max_label': load_volume(label_file),
                      'neighbors': load_volume(neighbor_file)}
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
    cspmax.setNumberOfSubjectsObjectsAndContrasts(1,structures,contrasts)
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
    
    if atlas_space is False and map_to_target is not None:
        print("map atlas to subject")
        print("load: "+str(map_to_target))
        mdata =  load_volume(map_to_target).get_data()
        cspmax.setMappingToTarget(nighresjava.JArray('float')(
                                            (mdata.flatten('F')).astype(float)))
        
    cspmax.setShapeAtlasProbasAndLabels(nighresjava.JArray('float')(
                                (pdata.flatten('F')).astype(float)),
                                nighresjava.JArray('int')(
                                (ldata.flatten('F')).astype(int).tolist()))

    if atlas_space is True and map_to_atlas is not None and map_to_target is not None:
        print("map subject to atlas")
        print("load: "+str(map_to_atlas))
        mdata =  load_volume(map_to_atlas).get_data()
        cspmax.setMappingToAtlas(nighresjava.JArray('float')(
                                            (mdata.flatten('F')).astype(float)))
        print("load: "+str(map_to_target))
        mdata =  load_volume(map_to_target).get_data()
        cspmax.setMappingToTarget(nighresjava.JArray('float')(
                                            (mdata.flatten('F')).astype(float)))

    # execute
    try:
        cspmax.estimateTarget()
        cspmax.strictSimilarityDiffusion(ngb_size)
        #cspmax.fastSimilarityDiffusion(ngb_size)
        #cspmax.collapseConditionalMaps()
        #if adjust_volume:
        if atlas_space is True and map_to_atlas is not None and map_to_target is not None:
            #cspmax.mappedOptimalVolumeThreshold(2.0, 0.01, True)
            #cspmax.mappedOptimalVolumeThreshold(1.0, 0.5, True)
            # definitely too flat? or not enough?? cspmax.mappedOptimalVolumeThreshold(2.0, 0.1, True)
            cspmax.mappedOptimalVolumeCertaintyThreshold(1.0)
        else:    
            #cspmax.optimalVolumeThreshold(2.0, 0.01, True)
            #cspmax.optimalVolumeThreshold(1.0, 0.5, True)
            # definitely too flat? or not enough?? cspmax.optimalVolumeThreshold(2.0, 0.1, True)
            cspmax.conditionalVolumeCertaintyThreshold(2.0)
            #cspmax.conditionalVolumeCertaintyGrowth(1.0)
        #else:
        #    if atlas_space is True and map_to_atlas is not None and map_to_target is not None:
        #        cspmax.mappedOptimalCertaintyThreshold()
        #    else:    
        #        #cspmax.optimalVolumeThreshold(1.0, 0.05, True)
        #        cspmax.optimalCertaintyThreshold()
        cspmax.collapseSpatialPriorMaps()
        cspmax.collapseConditionalMaps()

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

    if atlas_space is False or (map_to_atlas is None and map_to_target is None):
        dims3D = dims3Dtrg
        dims_ngb = (trg_dimensions[0],trg_dimensions[1],trg_dimensions[2],ngb_size)

    intens_dims = (structures+1,structures+1,contrasts)

    intens_hist_dims = ((structures+1)*(structures+1),cspmax.getNumberOfBins()+4,contrasts)

    spatial_proba_data = np.reshape(np.array(cspmax.getBestSpatialProbabilityMaps(1),
                                   dtype=np.float32), dims3D, 'F')

    spatial_label_data = np.reshape(np.array(cspmax.getBestSpatialProbabilityLabels(1),
                                    dtype=np.int32), dims3D, 'F')    

    combined_proba_data = np.reshape(np.array(cspmax.getBestProbabilityMaps(1),
                                   dtype=np.float32), dims3D, 'F')

    combined_label_data = np.reshape(np.array(cspmax.getBestProbabilityLabels(1),
                                    dtype=np.int32), dims3D, 'F')

    proba_data = np.reshape(np.array(cspmax.getFinalProba(),
                                   dtype=np.float32), dims3Dtrg, 'F')

    label_data = np.reshape(np.array(cspmax.getFinalLabel(),
                                    dtype=np.int32), dims3Dtrg, 'F')

    neighbor_data = np.reshape(np.array(cspmax.getNeighborhoodMaps(ngb_size),
                                        dtype=np.float32), dims_ngb, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(spatial_proba_data)
    spatial_proba = nb.Nifti1Image(spatial_proba_data, affine, header)

    header['cal_max'] = np.nanmax(spatial_label_data)
    spatial_label = nb.Nifti1Image(spatial_label_data, affine, header)

    header['cal_max'] = np.nanmax(combined_proba_data)
    combined_proba = nb.Nifti1Image(combined_proba_data, affine, header)

    header['cal_max'] = np.nanmax(combined_label_data)
    combined_label = nb.Nifti1Image(combined_label_data, affine, header)

    trg_header['cal_max'] = np.nanmax(proba_data)
    proba = nb.Nifti1Image(proba_data, trg_affine, trg_header)

    trg_header['cal_max'] = np.nanmax(label_data)
    label = nb.Nifti1Image(label_data, trg_affine, trg_header)

    header['cal_min'] = np.nanmin(neighbor_data)
    header['cal_max'] = np.nanmax(neighbor_data)
    neighbors = nb.Nifti1Image(neighbor_data, affine, header)

    if save_data:
        save_volume(spatial_proba_file, spatial_proba)
        save_volume(spatial_label_file, spatial_label)
        save_volume(combined_proba_file, combined_proba)
        save_volume(combined_label_file, combined_label)
        save_volume(proba_file, proba)
        save_volume(label_file, label)
        save_volume(neighbor_file, neighbors)

    output= {'max_spatial_proba': spatial_proba, 'max_spatial_label': spatial_label, 
            'max_combined_proba': combined_proba, 'max_combined_label': combined_label, 
            'max_proba': proba, 'max_label': label, 'neighbors': neighbors}

    return output


def conditional_shape_atlasing(subjects, structures, contrasts,
                      levelset_images=None, contrast_images=None,
                      histograms=True, map_to_atlas=None,
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
    contrast_images: [niimg]
        Atlas images to use in the parcellation, indexed by (subjects, contrasts)
    histograms: bool
        Whether to use complete histograms for intensity priors (default is True)
    map_to_atlas: niimg
        Coordinate mapping from the target to the atlas (opt)
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
        * cond_mean (niimg): Conditional intensity mean (_cspmax-cmean)
        * cond_stdv (niimg): Conditional intensity stdv (_cspmax-cstdv)
        * cond_hist (niimg): Conditional intensity histograms (_cspmax-chist)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    print('\nConditional Shape Atlasing')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, contrast_images[0][0])

        spatial_proba_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                  rootfile=contrast_images[0][0],
                                  suffix='cspmax-sproba', ))

        spatial_label_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=contrast_images[0][0],
                                   suffix='cspmax-slabel'))

        condmean_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                  rootfile=contrast_images[0][0],
                                  suffix='cspmax-cmean', ))

        condstdv_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=contrast_images[0][0],
                                   suffix='cspmax-cstdv'))
        
        condhist_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=contrast_images[0][0],
                                   suffix='cspmax-chist'))
        
        if overwrite is False \
            and os.path.isfile(spatial_proba_file) \
            and os.path.isfile(spatial_label_file) \
            and ( (histograms and os.path.isfile(condhist_file)) \
            or (os.path.isfile(condmean_file) \
            and os.path.isfile(condstdv_file)) ):
            
            print("skip computation (use existing results)")
            output = {'max_spatial_proba': load_volume(spatial_proba_file), 
                      'max_spatial_label': load_volume(spatial_label_file)}
            if histograms:
                output.update(cond_hist=load_volume(condhist_file))
            else:
                output.update(cond_mean=load_volume(condmean_file)) 
                output.update(cond_stdv=load_volume(condstdv_file))
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

    if map_to_atlas is not None:
        img = load_volume(map_to_atlas[0])
        data = img.get_data()
        header = img.get_header()
        affine = img.get_affine()
        resolution = [x.item() for x in header.get_zooms()]
        dimensions = data.shape
        
        cspmax.setAtlasDimensions(dimensions[0], dimensions[1], dimensions[2])
        cspmax.setAtlasResolutions(resolution[0], resolution[1], resolution[2])
    else:
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
    if map_to_atlas is not None:
        print("map subjects to atlas")
        for sub in range(subjects):
            print("load: "+str(map_to_atlas[sub]))
            mdata =  load_volume(map_to_atlas[sub]).get_data()
            cspmax.setMappingImageAt(sub, nighresjava.JArray('float')(
                                            (mdata.flatten('F')).astype(float)))

    # execute
    try:
        #cspmax.execute()
        if map_to_atlas is not None: 
            cspmax.computeMappedAtlasPriors()
        else:
            cspmax.computeAtlasPriors()
 
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    dimensions = (dimensions[0],dimensions[1],dimensions[2],cspmax.getBestDimension())
    dims3Dtrg = (trg_dimensions[0],trg_dimensions[1],trg_dimensions[2])

    intens_dims = (structures+1,structures+1,contrasts)
    intens_hist_dims = ((structures+1)*(structures+1),cspmax.getNumberOfBins()+4,contrasts)

    spatial_proba_data = np.reshape(np.array(cspmax.getBestSpatialProbabilityMaps(dimensions[3]),
                                   dtype=np.float32), dimensions, 'F')

    spatial_label_data = np.reshape(np.array(cspmax.getBestSpatialProbabilityLabels(dimensions[3]),
                                    dtype=np.int32), dimensions, 'F')    

    if histograms:
        intens_hist_data = np.reshape(np.array(cspmax.getConditionalHistogram(),
                                       dtype=np.float32), intens_hist_dims, 'F')
    else:
        intens_mean_data = np.reshape(np.array(cspmax.getConditionalMean(),
                                       dtype=np.float32), intens_dims, 'F')
    
        intens_stdv_data = np.reshape(np.array(cspmax.getConditionalStdv(),
                                        dtype=np.float32), intens_dims, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(spatial_proba_data)
    spatial_proba = nb.Nifti1Image(spatial_proba_data, affine, header)

    header['cal_max'] = np.nanmax(spatial_label_data)
    spatial_label = nb.Nifti1Image(spatial_label_data, affine, header)

    if histograms:
        chist = nb.Nifti1Image(intens_hist_data, None, None)
    else:
        cmean = nb.Nifti1Image(intens_mean_data, None, None)
        cstdv = nb.Nifti1Image(intens_stdv_data, None, None)

    if save_data:
        save_volume(spatial_proba_file, spatial_proba)
        save_volume(spatial_label_file, spatial_label)
        if histograms:
            save_volume(condhist_file, chist)
        else:
            save_volume(condmean_file, cmean)
            save_volume(condstdv_file, cstdv)

    output= {'max_spatial_proba': spatial_proba, 'max_spatial_label': spatial_label}
    if histograms:
        output.update(cond_hist=chist)
    else:
        output.update(cond_mean=cmean) 
        output.update(cond_stdv=cstdv)
    return output
