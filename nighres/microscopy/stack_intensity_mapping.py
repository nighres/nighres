import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def stack_intensity_mapping(image, references, mapped, weights = None,
                            patch=2, search=3,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):
    """ Stack intensity mapping

    Uses a simple non-local means approach adapted from [1]_

    Parameters
    ----------
    image: niimg
        Input 2D image
    references: [niimg]
        Reference 2D images to use for intensity mapping
    mapped: [niimg]
        Corresponding mapped 2D images to use for intensity mapping
    weights: [float], optional
        Weight factors for the 2D images (default is 1 for all)
    patch: int, optional 
        Maximum distance to define patch size (default is 2)
    search: int, optional 
        Maximum distance to define search window size (default is 3)
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

        * result (niimg): The intensity mapped input

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.


    References
    ----------
    .. [1] P. Coupé, J.V. Manjón, V. Fonov, J. Pruessner, M. Robles, D.L. Collins,
       Patch-based segmentation using expert priors: Application to hippocampus 
       and ventricle msegmentation, NeuroImage, vol. 54, pp. 940--954, 2011.
    """

    print('\nStack Intensity Mapping')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        result_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=image,
                                   suffix='sim-img'))

        if overwrite is False \
            and os.path.isfile(result_file) :
                print("skip computation (use existing results)")
                output = {'result': result_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    sim = nighresjava.NonlocalIntensityMapping()

    # set parameters
    
    # load image and use it to set dimensions and resolution
    img = load_volume(image)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape

    sim.setDimensions(dimensions[0], dimensions[1], 1)
       
    sim.setInputImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    sim.setReferenceNumber(len(references))
    
    for idx,ref in enumerate(references):
        data = load_volume(ref).get_fdata()
        sim.setReferenceImageAt(idx,nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

        data = load_volume(mapped[idx]).get_fdata()
        sim.setMappedImageAt(idx,nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

        if weights is not None:
            sim.setWeightAt(idx, weights[idx])
        else:
            sim.setWeightAt(idx, 1.0)
            
    # set algorithm parameters
    sim.setPatchDistance(patch)
    sim.setSearchDistance(search)
    
    # execute the algorithm
    try:
        sim.execute2D()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    data = np.reshape(np.array(sim.getMappedImage(),
                                    dtype=np.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(data)
    header['cal_max'] = np.nanmax(data)
    result = nb.Nifti1Image(data, affine, header)

    if save_data:
        save_volume(result_file, result)
        return {'result': result_file}
    else:
        return {'result': result}
