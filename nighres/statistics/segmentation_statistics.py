import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir


def segmentation_statistics(segmentation, intensity=None, template=None,
                            statistics=None, output_csv=None,
                            atlas=None, skip_first=True, ignore_zero=True,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):
    """ Segmentation Statistics

    Compute various statistics of image segmentations

    Parameters
    ----------
    segmentation: niimg
        Input segmentation image
    intensity: niimg, optional
        Input intensity image for intensity-based statistics
    template: niimg, optional
        Input template segmentation for comparisons
    statistics: [str] 
        Statistics to compute. Available options include:
        "Voxels", "Volume", "Mean_intensity", "Std_intensity",
        "10_intensity","25_intensity","50_intensity","75_intensity","90_intensity",
		"Volumes", "Dice_overlap", "Jaccard_overlap", "Volume_difference",
        "False_positives","False_negatives",
        "Dilated_Dice_overlap","Dilated_false_positive","Dilated_false_negative",
        "Dilated_false_negative_volume","Dilated_false_positive_volume",
        "Detected_clusters", "False_detections",
        "Cluster_numbers", "Mean_cluster_sizes", "Cluster_maps",
        "Average_surface_distance", "Average_surface_difference", 
        "Average_squared_surface_distance", "Hausdorff_distance"
    output_csv: str
        File name of the statistics file to generate or expand
    atlas: str, optional
        File name of an atlas file defining the segmentation labels
    skip_first: bool
        Whether to skip the first segmentation label (usually representing the 
        background, default is True)
    ignore_zero: bool
        Whether to ignore zero intensity values in the intensity image 
        (default is True)
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

        * csv (str): The csv statistics file
        * map (niimg): Map of the estimated statistic, if relevant (opt)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    print('\nSegmentation statistics')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir,segmentation)

        map_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                   rootfile=segmentation,
                                   suffix='stat-map'))

        if overwrite is False \
            and os.path.isfile(map_file) :
                # check that the denoised data is the same too
                print("skip computation (use existing results)")
                output = {'csv': output_csv, 'map': load_volume(map_file)}
                return output

        if overwrite is True:
            # delete current stats file to start from the beginning
            os.remove(output_csv)

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='12000m', maxheap='12000m')
    except ValueError:
        pass
    # create algorithm instance
    stats = cbstools.StatisticsSegmentation()

    # load first image and use it to set dimensions and resolution
    img = load_volume(segmentation)
    data = img.get_data()
    affine = img.get_affine()
    header = img.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    stats.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    stats.setResolutions(resolution[0], resolution[1], resolution[2])

    stats.setSegmentationImage(cbstools.JArray('int')(
                                    (data.flatten('F')).astype(int)))
    stats.setSegmentationName(_fname_4saving(rootfile=segmentation))

    # other input images, if any
    if intensity is not None:
        data = load_volume(intensity).get_data()
        stats.setIntensityImage(cbstools.JArray('float')(
                                    (data.flatten('F')).astype(float)))
        stats.setIntensityName(_fname_4saving(rootfile=intensity))
    
    if template is not None:
        data = load_volume(template).get_data()
        stats.setTemplateImage(cbstools.JArray('int')(
                                    (data.flatten('F')).astype(int)))
        stats.setTemplateName(_fname_4saving(rootfile=template))
    
    # set algorithm parameters
    if atlas is not None:
        stats.setAtlasFile(atlas)
        
    stats.setSkipFirstLabel(skip_first)
    stats.setIgnoreZeroIntensities(ignore_zero)
    
    if len(statistics)>0: stats.setStatistic1(statistics[0])
    if len(statistics)>1: stats.setStatistic2(statistics[1])
    if len(statistics)>2: stats.setStatistic3(statistics[2])

    # execute the algorithm
    try:
        stats.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    boolean output=False
    for st in statistics: if st=="Cluster_maps": output=True
    
    if (phase_list!=None):
        for idx, image in enumerate(phase_list):
            den_data = np.reshape(np.array(lpca.getDenoisedPhaseImageAt(idx),
                                       dtype=np.int32), dimensions, 'F')
            header['cal_min'] = np.nanmin(den_data)
            header['cal_max'] = np.nanmax(den_data)
            denoised = nb.Nifti1Image(den_data, affine, header)
            denoised_list.append(denoised)
    
            if save_data:
                save_volume(den_files[idx+len(image_list)], denoised)

    dim_data = np.reshape(np.array(lpca.getLocalDimensionImage(),
                                    dtype=np.float32), dimensions, 'F')

    err_data = np.reshape(np.array(lpca.getNoiseFitImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(dim_data)
    header['cal_max'] = np.nanmax(dim_data)
    dim = nb.Nifti1Image(dim_data, affine, header)

    header['cal_min'] = np.nanmin(err_data)
    header['cal_max'] = np.nanmax(err_data)
    err = nb.Nifti1Image(err_data, affine, header)

    if save_data:
        save_volume(dim_file, dim)
        save_volume(err_file, err)

    return {'denoised': denoised_list, 'dimensions': dim, 'residuals': err}
