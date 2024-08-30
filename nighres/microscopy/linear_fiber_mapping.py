import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, \
                    _check_available_memory


def linear_fiber_mapping(input_image, ridge_intensities, 
                              min_scale=0, max_scale=3,
                              diffusion_factor=1.0,
                              similarity_scale=0.1,
                              max_iter=100, max_diff=1e-3,
                              threshold=0.5,
                              max_dist=1.0,
                              inclusion_ratio=0.1,
                              extend=False,
                              extend_ratio=0.5,
                              diameter=False,
                              save_data=False, overwrite=False, output_dir=None,
                              file_name=None):

    """ Linear Fiber Mapping 

    Extracts linear structures across multiple scales, and estimate their size
    and direction.


    Parameters
    ----------
    input_image: niimg
        Input image
    ridge_intensities: {'bright','dark','both'}
        Which intensities to consider for the filtering
    min_scale: int
        Minimum scale (in voxels) to look for features (default is 0)
    max_scale: int
        Maximum scale (in voxels) to look for features (default is 3)
    diffusion_factor: float
        Scaling factor for the diffusion weighting in [0,1] (default is 1.0)
    similarity_scale: float
        Scaling of the similarity function as a factor of intensity range
    max_iter: int
        Maximum number of diffusion iterations
    max_diff: int
        Maximum difference to stop the diffusion
    threshold: float
        Detection threshold for the structures to keep (default is 0.5)
    max_dist: float
        Maximum distance of voxels to include in lines (default is 1.0)
    inclusion_ratio: float
        Ratio of the highest detection value to include in lines (default is 0.1)
    extend: bool
        Whether or not to extend the estimation results into the background 
        and/or lower values (default is False)
    extend_ratio: float
        Ratio of the detection value to extend out (default is 0.5)
    diameter: bool
        Whether or not to estimate diameter and partial volume (default is False)
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

        * proba (niimg): propagated probabilistic response (_lfm-proba)
        * lines (niimg): labeling of individual lines (_lfm-lines)
        * length (niimg): estimated line length (_lfm-length)
        * theta (niimg): estimated line orientation angle (_lfm-theta)
        * ani (niimg): estimated line anisotropy (_lfm-ani)
        * dia (niimg): estimated local diameter (_lfm-dia)
        * pv (niimg): estimated local partial volume (_lfm-pv)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin. Based on the recursive ridge
    filter and vessel diameter estimation in [1]_.

    References
    ----------
    .. [1] Bazin et al (2016), Vessel segmentation from quantitative
           susceptibility maps for local oxygenation venography, Proc ISBI.

    """

    print('\n Linear Fiber Mapping')

    # check atlas_file and set default if not given
    #atlas_file = _check_atlas_file(atlas_file)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_image)

        proba_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='lfm-proba'))

        lines_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=input_image,
                                   suffix='lfm-lines'))

        length_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='lfm-length'))

        theta_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='lfm-theta'))

        ani_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='lfm-ani'))

        dia_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='lfm-dia'))

        pv_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=input_image,
                                  suffix='lfm-pv'))

        if overwrite is False \
            and os.path.isfile(proba_file) \
            and os.path.isfile(lines_file) \
            and os.path.isfile(length_file) \
            and os.path.isfile(theta_file) \
            and os.path.isfile(ani_file) :

            print("skip computation (use existing results)")
            output = {'proba': proba_file,
                      'lines': lines_file,
                      'length': length_file,
                      'theta': theta_file,
                      'ani': ani_file}
            return output


    # load input image and use it to set dimensions and resolution
    img = load_volume(input_image)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    if (len(dimensions)<3): dimensions = (dimensions[0], dimensions[1], 1)
    if (len(resolution)<3): resolution = [resolution[0], resolution[1], 1.0]

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create extraction instance
    if (dimensions[2]>1):
        lfm = nighresjava.LinearFiberMapping3D()
    else:
        lfm = nighresjava.LinearFiberMapping()

    # set parameters
    lfm.setRidgeIntensities(ridge_intensities)
    lfm.setMinimumScale(min_scale)
    lfm.setMaximumScale(max_scale)
    lfm.setDiffusionFactor(diffusion_factor)
    lfm.setSimilarityScale(similarity_scale)
    lfm.setMaxIterations(max_iter)
    lfm.setMaxDifference(max_diff)
    lfm.setDetectionThreshold(threshold)
    lfm.setMaxLineDistance(max_dist)
    lfm.setInclusionRatio(inclusion_ratio)
    lfm.setExtendResult(extend)
    lfm.setExtendRatio(extend_ratio)
    lfm.setEstimateDiameter(diameter)
    
    lfm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    lfm.setResolutions(resolution[0], resolution[1], resolution[2])

    # input input_image
    lfm.setInputImage(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))

    # execute Extraction
    try:
        lfm.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    if (dimensions[2]>1):
        dim4d = (dimensions[0], dimensions[1], dimensions[2], 3)
    else:
        dim4d = dimensions
        
    # reshape output to what nibabel likes
    proba_data = np.reshape(np.array(lfm.getProbabilityResponseImage(),
                                    dtype=np.float32), newshape=dimensions, order='F')

    lines_data = np.reshape(np.array(lfm.getLineImage(),
                                   dtype=np.int32), newshape=dimensions, order='F')

    length_data = np.reshape(np.array(lfm.getLengthImage(),
                                   dtype=np.float32), newshape=dimensions, order='F')

    theta_data = np.reshape(np.array(lfm.getAngleImage(),
                                    dtype=np.float32), newshape=dim4d, order='F')

    ani_data = np.reshape(np.array(lfm.getAnisotropyImage(),
                                    dtype=np.float32), newshape=dimensions, order='F')

    if diameter:
       dia_data = np.reshape(np.array(lfm.getDiameterImage(),
                                    dtype=np.float32), newshape=dimensions, order='F')
        
       pv_data = np.reshape(np.array(lfm.getPartialVolumeImage(),
                                    dtype=np.float32), newshape=dimensions, order='F')
        

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_max'] = np.nanmax(proba_data)
    proba_img = nb.Nifti1Image(proba_data, affine, header)

    header['cal_max'] = np.nanmax(lines_data)
    lines_img = nb.Nifti1Image(lines_data, affine, header)

    header['cal_max'] = np.nanmax(length_data)
    length_img = nb.Nifti1Image(length_data, affine, header)

    header['cal_max'] = np.nanmax(theta_data)
    theta_img = nb.Nifti1Image(theta_data, affine, header)

    header['cal_max'] = np.nanmax(ani_data)
    ani_img = nb.Nifti1Image(ani_data, affine, header)
    
    if diameter:
        header['cal_max'] = np.nanmax(dia_data)
        dia_img = nb.Nifti1Image(dia_data, affine, header)
        
        header['cal_max'] = np.nanmax(pv_data)
        pv_img = nb.Nifti1Image(pv_data, affine, header)
        

    if save_data:
        save_volume(proba_file, proba_img)
        save_volume(lines_file, lines_img)
        save_volume(length_file, length_img)
        save_volume(theta_file, theta_img)
        save_volume(ani_file, ani_img)
        if diameter:
            save_volume(dia_file, dia_img)
            save_volume(pv_file, pv_img)
            
            return {'proba': proba_file, 'lines': lines_file,
                'length': length_file, 'theta': theta_file,
                'ani': ani_file, 'dia': dia_file, 'pv': pv_file}
        else:
            return {'proba': proba_file, 'lines': lines_file,
                'length': length_file, 'theta': theta_file,
                'ani': ani_file}
    else:
        if diameter:
            return {'proba': proba_img, 'lines': lines_img,
                'length': length_img, 'theta': theta_img,
                'ani': ani_img, 'dia': dia_img, 'pv': pv_img}
        else:
            return {'proba': proba_img, 'lines': lines_img,
                'length': length_img, 'theta': theta_img,
                'ani': ani_img}
