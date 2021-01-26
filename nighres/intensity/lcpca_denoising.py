import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def lcpca_denoising(image_list, phase_list=None, 
                    ngb_size=4, stdev_cutoff=1.05,
                    min_dimension=0, max_dimension=-1,
                    unwrap=True, rescale_phs=True, process_2d=False, use_rmt=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_names=None):
    """ LCPCA denoising

    Denoise multi-contrast data with a local complex-valued PCA-based method

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
    unwrap: bool, optional
        Whether to unwrap the phase data of keep it as is
        (default is True)
    rescale_phs: bool, optional
        Whether to rescale the phase data of keep it as is, assuming radians
        (default is True)
    process_2d: bool, optional
        Whether to denoise in 2D, for instance when acquiring a thin slab of 
        data (default is False)
    use_rmt: bool, optional
        Whether to use random matrix theory rather than noise fitting to
        estimate the noise threshold (default is False)
    save_data: bool
        Save output data to file (default is False)
    overwrite: bool
        Overwrite existing results (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_names: [str], optional
        Desired base names for output files with file extension
        (suffixes will be added)

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * denoised ([niimg]): The list of denoised input images (_lcpca_den)
        * dimensions (niimg): Map of the estimated local dimensions (_lcpca_dim)
        * residuals (niimg): Estimated residuals between input and denoised images (_lcpca_err)

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

    print('\nLCPCA denoising')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image_list[0])

        den_files = []
        for idx,image in enumerate(image_list):
            if file_names is None: name=None
            else: name=file_names[idx]
            den_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=name,
                                      rootfile=image,
                                      suffix='lcpca-den'))
            den_files.append(den_file)

        if (phase_list!=None):
            for idx,image in enumerate(phase_list):
                if file_names is None: name=None
                else: name=file_names[len(image_list)+idx]
                den_file = os.path.join(output_dir,
                            _fname_4saving(module=__name__,file_name=name,
                                          rootfile=image,
                                          suffix='lcpca-den'))
                den_files.append(den_file)

        if file_names is None: name=None
        else: name=file_names[0]
        dim_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=name,
                                   rootfile=image_list[0],
                                   suffix='lcpca-dim'))

        err_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=name,
                                   rootfile=image_list[0],
                                   suffix='lcpca-res'))
        
        if overwrite is False \
            and os.path.isfile(dim_file) \
            and os.path.isfile(err_file) :
                # check that the denoised data is the same too
                missing = False
                for den_file in den_files:
                    if not os.path.isfile(den_file):
                        missing = True
                if not missing:
                    print("skip computation (use existing results)")
                    denoised = []
                    for den_file in den_files:
                        denoised.append(den_file)
                    output = {'denoised': denoised,
                              'dimensions': dim_file,
                              'residuals': err_file}
                    
                    return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create lcpca instance
    lcpca = nighresjava.LocalComplexPCADenoising()

    # load first image and use it to set dimensions and resolution
    img = load_volume(image_list[0])
    data = img.get_data()
    #data = data[0:10,0:10,0:10]
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    dim3D = (dimensions[0],dimensions[1],dimensions[2])
    
    # set lcpca parameters
    lcpca.setImageNumber(len(image_list))
    eigdim = len(image_list)
    if (phase_list!=None): 
        if len(dimensions)>3:
            eigdim = 2*eigdim*dimensions[3]
        else:
            eigdim = 2*eigdim
        if (len(phase_list)!=len(image_list)):
            print('\nmismatch of magnitude and phase images: abort')
            return

    if len(dimensions)>3:
        lcpca.setDimensions(dimensions[0], dimensions[1], dimensions[2], dimensions[3])
    else:
        lcpca.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    lcpca.setResolutions(resolution[0], resolution[1], resolution[2])

    # input images
    # important: set image number before adding images
    for idx, image in enumerate(image_list):
            #print('\nloading ('+str(idx)+'): '+image)
            data = load_volume(image).get_data()
            #data = data[0:10,0:10,0:10]
            lcpca.setMagnitudeImageAt(idx, nighresjava.JArray('float')(
                                        (data.flatten('F')).astype(float)))

    # input phase, if specified
    if (phase_list!=None):
        for idx, image in enumerate(phase_list):
            #print('\nloading '+image)
            data = load_volume(image).get_data()
            #data = data[0:10,0:10,0:10]
            lcpca.setPhaseImageAt(idx, nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

    # set algorithm parameters
    lcpca.setPatchSize(ngb_size)
    lcpca.setStdevCutoff(stdev_cutoff)
    lcpca.setMinimumDimension(min_dimension)
    lcpca.setMaximumDimension(max_dimension)
    lcpca.setUnwrapPhase(unwrap) 
    lcpca.setRescalePhase(rescale_phs) 
    lcpca.setProcessSlabIn2D(process_2d)
    lcpca.setRandomMatrixTheory(use_rmt)

    # execute the algorithm
    try:
        lcpca.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    denoised_list = []
    for idx, image in enumerate(image_list):
        den_data = np.reshape(np.array(lcpca.getDenoisedMagnitudeImageAt(idx),
                                   dtype=np.float32), dimensions, 'F')
        header['cal_min'] = np.nanmin(den_data)
        header['cal_max'] = np.nanmax(den_data)
        denoised = nb.Nifti1Image(den_data, affine, header)
        denoised_list.append(denoised)

        if save_data:
            save_volume(den_files[idx], denoised)

    if (phase_list!=None):
        for idx, image in enumerate(phase_list):
            den_data = np.reshape(np.array(lcpca.getDenoisedPhaseImageAt(idx),
                                       dtype=np.float32), dimensions, 'F')
            header['cal_min'] = np.nanmin(den_data)
            header['cal_max'] = np.nanmax(den_data)
            denoised = nb.Nifti1Image(den_data, affine, header)
            denoised_list.append(denoised)

            if save_data:
                save_volume(den_files[idx+len(image_list)], denoised)

    dim_data = np.reshape(np.array(lcpca.getLocalDimensionImage(),
                                    dtype=np.float32), dim3D, 'F')

    err_data = np.reshape(np.array(lcpca.getNoiseFitImage(),
                                    dtype=np.float32), dim3D, 'F')

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
        output = {'denoised': den_files, 'dimensions': dim_file, 'residuals': err_file}
    else:
        output = {'denoised': denoised_list, 'dimensions': dim, 'residuals': err}

    return output
