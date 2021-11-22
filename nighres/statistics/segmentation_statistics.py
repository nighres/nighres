import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


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
        "Voxels", "Volume", "Center_of_mass", "Mean_intensity",
        "Std_intensity", "Sum_intensity", "10_intensity","25_intensity",
        "50_intensity", "75_intensity","90_intensity", "Median_intensity",
        "IQR_intensity", "SNR_intensity","rSNR_intensity", "Volumes", 
        "Dice_overlap", "Jaccard_overlap", "Volume_difference", "False_positives"
        "False_negatives", "Dilated_Dice_overlap","Dilated_false_positive",
        "Dilated_false_negative", "Dilated_false_negative_volume",
        "Dilated_false_positive_volume", "Center_distance", "Detected_clusters",
        "False_detections", "Cluster_numbers", "Mean_cluster_sizes",
        "Cluster_maps", "Average_surface_distance",
        "Average_surface_difference", "Average_squared_surface_distance",
        "Hausdorff_distance"
    output_csv: str
        File name of the statistics file to generate or expand
    atlas: str, optional
        File name of an atlas file defining the segmentation labels
    skip_first: bool, optional
        Whether to skip the first segmentation label (usually representing the
        background, default is True)
    ignore_zero: bool, optional
        Whether to ignore zero intensity values in the intensity image
        (default is True)
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

        * csv (str): The csv statistics file
        * map (niimg): Map of the estimated statistic, if relevant (stat-map)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    print('\nSegmentation statistics')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir,segmentation)

        map_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=segmentation,
                                   suffix='stat-map'))

        csv_file = os.path.join(output_dir, output_csv)

        if overwrite is False \
            and os.path.isfile(csv_file) :
                # check that the denoised data is the same too
                print("append results to existing csv file")

        if overwrite is True:
            # delete current stats file to start from the beginning
            os.remove(csv_file)
    else:
        csv_file = output_csv

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    stats = nighresjava.StatisticsSegmentation()

    # load first image and use it to set dimensions and resolution
    img = load_volume(segmentation)
    data = img.get_data()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    if len(dimensions)>2:
        stats.setDimensions(dimensions[0], dimensions[1], dimensions[2])
        stats.setResolutions(resolution[0], resolution[1], resolution[2])
    else:
        stats.setDimensions(dimensions[0], dimensions[1], 1)
        stats.setResolutions(resolution[0], resolution[1], 1.0)
        
    stats.setSegmentationImage(nighresjava.JArray('int')(
                                    (data.flatten('F')).astype(int).tolist()))
    stats.setSegmentationName(_fname_4saving(module=__name__,rootfile=segmentation))

    # other input images, if any
    if intensity is not None:
        data = load_volume(intensity).get_data()
        stats.setIntensityImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
        stats.setIntensityName(_fname_4saving(module=__name__,rootfile=intensity))

    if template is not None:
        data = load_volume(template).get_data()
        stats.setTemplateImage(nighresjava.JArray('int')(
                                    (data.flatten('F')).astype(int).tolist()))
        stats.setTemplateName(_fname_4saving(module=__name__,rootfile=template))

    # set algorithm parameters
    if atlas is not None:
        stats.setAtlasFile(atlas)

    stats.setSkipFirstLabel(skip_first)
    stats.setIgnoreZeroIntensities(ignore_zero)

    stats.setStatisticNumber(len(statistics))
    for idx,stat in enumerate(statistics): stats.setStatisticAt(idx, stat)

    stats.setSpreadsheetFile(csv_file)

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
    output = False
    for st in statistics:
        if st=="Cluster_maps":
            output=True

    if (output):
        data = np.reshape(np.array(stats.getOutputImage(),
                                       dtype=np.int32), dimensions, 'F')
        header['cal_min'] = np.nanmin(data)
        header['cal_max'] = np.nanmax(data)
        output = nb.Nifti1Image(data, affine, header)

        if save_data:
            save_volume(map_file, output)

    csv_file = stats.getOutputFile()

    if output:
        if save_data:
            return {'csv': csv_file, 'map': map_file}
        else:
            return {'csv': csv_file, 'map': output}
    else:
        return {'csv': csv_file}
