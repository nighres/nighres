import os
import sys
import numpy
import nibabel
import nighresjava
from ..io import load_volume, save_volume, load_mesh, save_mesh
from ..utils import _output_dir_4saving, _fname_4saving,_check_available_memory


def parcellation_smoothing(parcellation, probability=None, connectivity="wcs", 
                     smoothing=1.0,
                     save_data=False, overwrite=False,
                     output_dir=None, file_name=None):

    """Parcellation smoothing

    Make smooth boundaries for noisy parcellations using a MGDM [1]_ algorithm
    to regularize curvature.

    Parameters
    ----------
    parcellation: niimg
        Parcellation image to be regularized
    probability: niimg
        Probability image associated to the parcellation (certainty of the parcellation)
    connectivity: {"6/18","6/26","18/6","26/6"}, optional
        Choice of digital connectivity to build the mesh (default is 18/6)
    smoothing: float, optional
        Smoothing of the boundary, high values may bring distortions (default is 1.0)
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

        * label (niimg): Smoothed parcellation labels (_smooth-label)
        * proba (niimg): Smoothed parcellation probabilities (_smooth-proba)
       
    Notes
    ----------
    Ported from original Java module by Pierre-Louis Bazin. Original algorithm
    from [1]_.

    References
    ----------
    .. [1] Bogovic et al. (2013). A multiple object geometric deformable model 
       for image segmentation.
       doi:10.1016/j.cviu.2012.10.006.A
    """

    print("\nParcellation smoothing")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, parcellation)

        proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=parcellation,
                                  suffix='smooth-proba', ))

        label_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=parcellation,
                                   suffix='smooth-label'))

        if overwrite is False \
            and os.path.isfile(proba_file) \
            and os.path.isfile(label_file):
            
            print("skip computation (use existing results)")
            output = {'proba': proba_file, 
                      'label': label_file}
            return output

    # start virtual machine if not running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    
    # initiate class inside the loop, to avoid garbage collection issues with many labels (??)
    algorithm = nighresjava.ParcellationMgdmSmoothing()

    # first we need to know how many meshes to build
    
    # load the data
    img = load_volume(parcellation)
    data = img.get_fdata()
    hdr = img.header
    aff = img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = data.shape

    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])
    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])

    algorithm.setParcellationImage(nighresjava.JArray('int')(
                                (data.flatten('F')).astype(int).tolist()))

    if probability is not None:
        data = load_volume(probability).get_fdata()
        algorithm.setProbabilityImage(nighresjava.JArray('float')(
                                            (data.flatten('F')).astype(float)))

    algorithm.setTopology(connectivity)
    algorithm.setCurvatureWeight(0.5*smoothing/(smoothing+1.0))
    algorithm.setDataWeight(0.5/(smoothing+1.0))
    algorithm.setMaxIterations(500)
    algorithm.setMinChange(0.001)
    
    # execute class
    try:
        algorithm.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return
    
    # collect outputs
    label_data = numpy.reshape(numpy.array(massp.getSmoothedLabel(),
                                    dtype=numpy.int32), dimensions, 'F')

    proba_data = numpy.reshape(numpy.array(massp.getSmoothedProba(),
                                    dtype=numpy.float32), dimensions, 'F')

    
    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    hdr['cal_max'] = numpy.nanmax(proba_data)
    proba = nibabel.Nifti1Image(proba_data, aff, hdr)

    hdr['cal_max'] = numpy.nanmax(label_data)
    label = nibabel.Nifti1Image(label_data, aff, hdr)

    if save_data:
        save_volume(proba_file, proba)
        save_volume(label_file, label)

        output= {'proba': proba_file, 'label': label_file}
        return output
    else:
        output= {'proba': proba, 'label': label}
        return output

