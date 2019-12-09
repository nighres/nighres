# basic dependencies
import os
import sys

# main dependencies: numpy, nibabel
import numpy as np
import nibabel as nb

# nighresjava and nighres functions
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
    _check_topology_lut_dir, _check_atlas_file, _check_available_memory


def multiscale_vessel_filter(input_image,
			prior_image=None,
                        structure_intensity='bright',
                        filterType = 'RRF',
			propagationtype = 'diffusion',
			threshold = 0.5,
			factor = 0.5,
			max_diff = 0.001,
			max_itr = 100,
			scale_step = 1.0,
			scale = 4,
                        invert_prior=False,
                        save_data=False, 
                        overwrite=False, 
                        output_dir=None,
                        file_name=None):

    """ Vessel filter with prior
    
    Uses an image filter to make a probabilistic image of ridge
    structures.


    Parameters
    ----------
    input_image: niimg
        Image containing structure-of-interest
    structure_intensity: str
        Image intensity of structure-of-interest 'bright', 'dark', or 'both'.
    filterType: str
	Decide for a filter type: either RRF or Hessian
    propagationtype: str
	Set the diffusion model of the filter: ither diffusion or believ propagation model
    threshold: float
	Set the propability treshold to decide at what probability the detected structure should be seen as a vessel, standard value is 0.5
    factor: float
	Diffusion factor between 0 and 100, standard value is 0.5
    max_diff: float
	maximal difference between 0 and 1, standard value is 0.001
    max_itr: int
	maximale iteration number between 0 and 100, standard value is 100
    scale_step: float
	scale step is between 0.25 and 2, stanard value is 1
    scale: int
	Scale number between 1 and 10, standard value is 4
    invert_prior: boolean, optional (default is False)
 	In case there is a prior, the prior can be considered as negative prior (False) or as positive prior (True)
    prior_image: niiimg, optional
	prior image, has to be a binary mask
    save_data: bool, optional
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
   segmentation
   filtered
   probability
   scale
   diameter
   length
   pv
   label
   direction

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin and Julia Huck.
    """

    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_image)

        vesselImage_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='seg', )

        filterImage_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='filtered')

        probaImage_file = _fname_4saving(file_name=file_name,
                                   rootfile=input_image,
                                   suffix='proba')

        scaleImage_file = _fname_4saving(file_name=file_name,
                                   rootfile=input_image,
                                   suffix='scale')

        diameterImage_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='dia', )

        lengthImage_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='length')

        pvImage_file = _fname_4saving(file_name=file_name,
                                   rootfile=input_image,
                                   suffix='pv')

        labelImage_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='label')

        directionImage_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='direction')

        if overwrite is False \
            and os.path.isfile(vesselImage_file) \
            and os.path.isfile(filterImage_file) \
            and os.path.isfile(probaImage_file) \
            and os.path.isfile(scaleImage_file) \
            and os.path.isfile(diameterImage_file) \
            and os.path.isfile(pvImage_file) \
            and os.path.isfile(lengthImage_file) \
            and os.path.isfile(labelImage_file) \
	    and os.path.isfile(directionImage_file) :
                output = {'segmentation': load_volume(vesselImage_file),
                          'filtered': load_volume(filterImage_file),
                          'probability': load_volume(probaImage_file),
                          'scale': load_volume(scaleImage_file),
                          'diameter': load_volume(diameterImage_file),
                          'pv': load_volume(pvImage_file),
                          'length':load_volume(lengthImage_file),
                          'label':load_volume(labelImage_file),
			  'diection':load_volume(directionImage_file)}
                return output

       


    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    vessel_filter = nighresjava.MultiscaleVesselFilter()

    # set parameters
    vessel_filter.setStructureIntensity(structure_intensity)
    vessel_filter.setFilterShape(filterType)
    vessel_filter.setThreshold(threshold)
    vessel_filter.setScaleStep(scale_step)
    vessel_filter.setScaleNumber(scale)
    vessel_filter.setPropagationModel(propagationtype)
    vessel_filter.setDiffusionFactor(factor)
    vessel_filter.setMaxDiff(max_diff)
    vessel_filter.setMaxItr(max_itr)
    vessel_filter.setInvertPrior(invert_prior)


    # load images and set dimensions and resolution
    input_image = load_volume(input_image)
    data = input_image.get_data()
    affine = input_image.get_affine()
    header = input_image.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = input_image.shape
    
    # direction output has a 4th dimension, set to 3
    dimensions4d = [dimensions[0], dimensions[1], dimensions[2], 3] 

    vessel_filter.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    vessel_filter.setResolutions(resolution[0], resolution[1], resolution[2])

    data = load_volume(input_image).get_data()
    vessel_filter.setInputImage(nighresjava.JArray('float')(
                               (data.flatten('F')).astype(float)))

    if not (prior_image==None):
        prior = load_volume(prior_image)
        data_prior = prior.get_data()
        vessel_filter.setPriorImage(nighresjava.JArray('float')((data_prior.flatten('F')).astype(float)))

    # execute
    try:
        vessel_filter.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # Collect output
    vesselImage_data = np.reshape(np.array(
                                    vessel_filter.getSegmentedVesselImage(),
                                    dtype=np.float32), dimensions, 'F')
    filterImage_data = np.reshape(np.array(
                                    vessel_filter.getFilteredImage(),
                                    dtype=np.float32), dimensions, 'F')
    probaImage_data = np.reshape(np.array(
                                    vessel_filter.getProbabilityImage(),
                                    dtype=np.float32), dimensions, 'F')
    scaleImage_data = np.reshape(np.array(
                                    vessel_filter.getScaleImage(),
                                    dtype=np.float32), dimensions, 'F')
    diameterImage_data = np.reshape(np.array(
                                    vessel_filter.getDiameterImage(),
                                    dtype=np.float32), dimensions, 'F')
    pvImage_data = np.reshape(np.array(
                                    vessel_filter.getPVimage(),
                                    dtype=np.float32), dimensions, 'F')
    lengthImage_data = np.reshape(np.array(
                                    vessel_filter.getLengthImage(),
                                    dtype=np.float32), dimensions, 'F')
    labelImage_data = np.reshape(np.array(
                                    vessel_filter.getLabelImage(),
                                    dtype=np.float32), dimensions, 'F')
    directionImage_data = np.reshape(np.array(
                                    vessel_filter.getDirectionImage(),
                                    dtype=np.float32), dimensions4d, 'F')


    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(vesselImage_data)
    vesselImage = nb.Nifti1Image(vesselImage_data, affine, header)

    header['cal_max'] = np.nanmax(filterImage_data)
    filterImage = nb.Nifti1Image(filterImage_data, affine, header)

    header['cal_max'] = np.nanmax(probaImage_data)
    probaImage = nb.Nifti1Image(probaImage_data, affine, header)

    header['cal_max'] = np.nanmax(scaleImage_data)
    scaleImage = nb.Nifti1Image(scaleImage_data, affine, header)

    header['cal_max'] = np.nanmax(diameterImage_data)
    diameterImage = nb.Nifti1Image(diameterImage_data, affine, header)

    header['cal_max'] = np.nanmax(pvImage_data)
    pvImage = nb.Nifti1Image(pvImage_data, affine, header)

    header['cal_max'] = np.nanmax(lengthImage_data)
    lengthImage = nb.Nifti1Image(lengthImage_data, affine, header)

    header['cal_max'] = np.nanmax(labelImage_data)
    labelImage = nb.Nifti1Image(labelImage_data, affine, header)

    header['cal_max'] = np.nanmax(directionImage_data)
    directionImage = nb.Nifti1Image(directionImage_data, affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, vesselImage_file), vesselImage)
        save_volume(os.path.join(output_dir, filterImage_file), filterImage)
        save_volume(os.path.join(output_dir, probaImage_file), probaImage)
        save_volume(os.path.join(output_dir, scaleImage_file), scaleImage)
        save_volume(os.path.join(output_dir, diameterImage_file), diameterImage)
        save_volume(os.path.join(output_dir, pvImage_file), pvImage)
        save_volume(os.path.join(output_dir, lengthImage_file), lengthImage)
        save_volume(os.path.join(output_dir, labelImage_file), labelImage)
        save_volume(os.path.join(output_dir, directionImage_file), directionImage)

    return {'segmentation': vesselImage, 'filtered': filterImage,
            'probability': probaImage, 'scale': scaleImage,
            'diameter': diameterImage, 'pv': pvImage, 
	    'length':lengthImage, 'label':labelImage, 'direction':directionImage}
