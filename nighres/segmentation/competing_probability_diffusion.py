import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def competing_probability_diffusion(probas, prior,
                            ratio=0.1, neighbors=4, maxdiff=0.01, maxiter=100,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):
    """ Competing probability diffusion

    Uses a simple diffusion probabilistic model to cluster detected features into labelled maps

    Parameters
    ----------
    probas: [niimg]
        Input images with probability for the labels
    prior: niimg
        Input image with unlabelled detected features 
    ratio: float
        Diffusion ratio in [0,1] (default is 0.1)
    neighbors: float
        Number of local neighbors to use (default is 4)
    maxdiff: float
        Maximum difference between iterations before stopping
    maxiter: int
        Maximum number of iterations
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

        * posteriors (niimg): The estimated posterior for each label
        * clustering (niimg): The estimated maximum probability clusters of detected features

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    
    """

    print('\nCompeting Probability Diffusion')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, prior)

        post_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=prior,
                                   suffix='cpd-post'))

        class_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=prior,
                                   suffix='cpd-class'))

        if overwrite is False \
            and os.path.isfile(post_file) and os.path.isfile(class_file) :
                print("skip computation (use existing results)")
                output =  {'posteriors': post_file, 'clustering': class_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    cpd = nighresjava.CompetingProbabilityDiffusion()

    # set parameters
    
    nimg = len(probas)
    cpd.setImageNumber(nimg)
    
    # load image and use it to set dimensions and resolution
    img = load_volume(probas[0])
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    dim4d = (dimensions[0], dimensions[1], dimensions[2], nimg)
    
    cpd.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    
    for idx,proba in enumerate(probas):
        data = load_volume(proba).get_fdata()
        cpd.setProbaImageAt(idx, nighresjava.JArray('float')(
                                (data.flatten('F')).astype(float)))
    
    data = load_volume(prior).get_fdata()
    cpd.setPriorImage(nighresjava.JArray('float')(
                                (data.flatten('F')).astype(float)))
    
    # set algorithm parameters
    cpd.setDiffusionRatio(ratio)
    cpd.setNeighborhoodSize(neighbors)
    cpd.setMaxIterations(maxiter)
    cpd.setMaxDifference(maxdiff)
    
    # execute the algorithm
    try:
        cpd.execute()
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    data = np.reshape(np.array(cpd.getPosteriorImages(),
                                    dtype=np.float32), dim4d, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(data)
    header['cal_max'] = np.nanmax(data)
    post_res = nb.Nifti1Image(data, affine, header)

    # reshape output to what nibabel likes
    data = np.reshape(np.array(cpd.getClusteringImage(),
                                    dtype=np.int32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(data)
    header['cal_max'] = np.nanmax(data)
    class_res = nb.Nifti1Image(data, affine, header)

    if save_data:
        save_volume(post_file, post_res)
        save_volume(class_file, class_res)
        return {'posteriors': post_file, 'clustering': class_file}
    else:
        return {'posteriors': post_res, 'clustering': class_res}
