import numpy as np
import nibabel as nb
import os
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir


def lpca_denoising(image_list, phase_list=None, 
                    ngb_size=4, stdev_cutoff=1.05,
                      min_dimension=0, max_dimension=-1,
                      save_data=False, output_dir=None,
                      file_name=None):
    """ LPCA denoising

    Denoise multi-contrast data with a local PCA-based method

    Parameters
    ----------
    image_list: [niimg]
        List of input images to denoise
    phase_list: [niimg], optional
        List of input phase to denoise (order must match that of image_list)
    ngb_size: int, optional
        Size of the local PCA neighborhood, to be increased with number of 
        inputs (default is 4)
    stdev_cutoff: float, optional 
        Factor of local noise level to remove PCA components. Higher
        values remove more components (default is 1.05)
    min_dimension: int, optional
        Minimum number of kept PCA components
        (default is 0)
    max_dimension: int, optional
        Maximum number of kept PCA components
        (default is -1 for all components)
    save_data: bool
        Save output data to file (default is False)
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

        * denoised ([niimg]): The list of denoised input images (_lpca_den)
        * dimensions (niimg): Map of the estimated local dimensions (_lpca_dim)
        * residuals (niimg): Estimated residuals between input and denoised
            images (_lpca_err)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Algorithm adapted from [1]_
    with a different approach to set the adaptive noise threshold and additional
    processing to handle the phase data.

    References
    ----------
    .. [1] Manjon, Coupe, Concha, Buades, Collins, Robles (2013). Diffusion 
        Weighted Image Denoising Using Overcomplete Local PCA
        doi:10.1371/journal.pone.0073021
    """

    print('\nLPCA denoising')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image_list[0])

        den_files = []
        for idx,image in enumerate(image_list):
            den_file = _fname_4saving(file_name=file_name,
                                      rootfile=image,
                                      suffix='lpca_den')
            den_files.append(den_file)

        if (phase_list!=None):
            for idx,image in enumerate(phase_list):
                den_file = _fname_4saving(file_name=file_name,
                                          rootfile=image,
                                          suffix='lpca_den')
                den_files.append(den_file)

        dim_file = _fname_4saving(file_name=file_name,
                                   rootfile=image_list[0],
                                   suffix='lpca_dim')

        err_file = _fname_4saving(file_name=file_name,
                                   rootfile=image_list[0],
                                   suffix='lpca_err')

    # start virtual machine, if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass
    # create lpca instance
    lpca = cbstools.IntensityComplexPCADenoising()

    # set lpca parameters
    lpca.setImageNumber(len(image_list))
    if (phase_list!=None): 
        if (len(phase_list)!=len(image_list)):
            print('\nmismatch of magnitude and phase images: abort')
            return
    
    # load first image and use it to set dimensions and resolution
    img = load_volume(image_list[0])
    data = img.get_data()
    #data = data[0:10,0:10,0:10]
    affine = img.get_affine()
    header = img.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    lpca.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    lpca.setResolutions(resolution[0], resolution[1], resolution[2])

    # input images
    # important: set image number before adding images
    for idx, image in enumerate(image_list):
        #print('\nloading ('+str(idx)+'): '+image)
        data = load_volume(image).get_data()
        #data = data[0:10,0:10,0:10]
        lpca.setMagnitudeImageAt(idx, cbstools.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    # input phase, if specified
    if (phase_list!=None):
        for idx, image in enumerate(phase_list):
            #print('\nloading '+image)
            data = load_volume(image).get_data()
            #data = data[0:10,0:10,0:10]
            lpca.setPhaseImageAt(idx, cbstools.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    # set algorithm parameters
    lpca.setPatchSize(ngb_size)
    lpca.setStdevCutoff(stdev_cutoff)
    lpca.setMinimumDimension(min_dimension)
    lpca.setMaximumDimension(max_dimension)
     

    # execute the algorithm
    try:
        lpca.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

    # reshape output to what nibabel likes
    denoised_list = []
    for idx, image in enumerate(image_list):
        den_data = np.reshape(np.array(lpca.getDenoisedMagnitudeImageAt(idx),
                                   dtype=np.int32), dimensions, 'F')
        header['cal_min'] = np.nanmin(den_data)
        header['cal_max'] = np.nanmax(den_data)
        denoised = nb.Nifti1Image(den_data, affine, header)
        denoised_list.append(denoised)

        if save_data:
            save_volume(os.path.join(output_dir, den_files[idx]), denoised)

    if (phase_list!=None):
        for idx, image in enumerate(phase_list):
            den_data = np.reshape(np.array(lpca.getDenoisedPhaseImageAt(idx),
                                       dtype=np.int32), dimensions, 'F')
            header['cal_min'] = np.nanmin(den_data)
            header['cal_max'] = np.nanmax(den_data)
            denoised = nb.Nifti1Image(den_data, affine, header)
            denoised_list.append(denoised)
    
            if save_data:
                save_volume(os.path.join(output_dir, 
                                    den_files[idx+len(image_list)]), denoised)

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
        save_volume(os.path.join(output_dir, dim_file), dim)
        save_volume(os.path.join(output_dir, err_file), err)

    return {'denoised': denoised_list, 'dimensions': dim, 'residuals': err}
