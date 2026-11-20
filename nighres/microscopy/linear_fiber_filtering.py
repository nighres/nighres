import numpy as np
import nibabel as nb
import os
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def linear_fiber_filtering(pv, diameter, theta, length,
                            labeling=None, clusters=0,
                            thickness=[0.0,5.0], angle=[30.0,60.0], size=[3.0,200.0],
                            smooth=0.0,scale=10.0,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):
    """ Linear fiber filtering

    Simple fltering of detected lines (from the linear_fiber_mapping module)
    into subclasses in 2D images

    Parameters
    ----------
    pv: niimg
        Input 2D image with partial volume of the detected lines
    diameter: niimg
        Input 2D images with diameter of the detected lines
    theta: niimg
        Input 2D images with angle of the detected lines
    length: niimg
        Input 2D images with length of the detected lines
    labeling: niimg, optional
        Reference 2D images with anatomical labels to define directions 
        (default is None, for constant directions)
    clusters: int
        Number of local clusters to define in each region, if using clustering
    thickness: [float]
        Thickness groups for associated lines
    angle: [float]
        Angle in degrees to expect between associated lines and underlying anatomy 
    size: [float]
        Size groups for associated lines
    smooth: float
        Smoothing ratio for labeling surfaces to improve angle definition (default is 0.0)
    scale: float
        Smoothing scale for labeling surfaces to improve angle definition (default is 0.0)
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

        * proba (niimg): The estimated likelihood of lines for their group
        * label (niimg): The estimated grouping of the lines

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    
    """

    print('\nLinear Fiber Filtering')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, pv)

        proba_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=pv,
                                   suffix='lff-proba'))

        label_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=pv,
                                   suffix='lff-label'))

        if overwrite is False \
            and os.path.isfile(proba_file) and os.path.isfile(label_file) :
                print("skip computation (use existing results)")
                output =  {'proba': proba_file, 'label': label_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    llf = nighresjava.LinearFiberFiltering()

    # set parameters

	# load image and use it to set dimensions and resolution
    img = load_volume(pv)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = data.shape
    if (len(dimensions)==2 or dimensions[2]==1):
        print("2D version")
        llf.setDimensions(dimensions[0], dimensions[1], 1)
    else:
        print("3D version")
        llf.setDimensions(dimensions[0], dimensions[1], dimensions[2])

    llf.setPartialVolumeImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

    # load other image files
    data = load_volume(diameter).get_fdata()
    llf.setDiameterImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    data = load_volume(theta).get_fdata()
    llf.setAngleImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    data = load_volume(length).get_fdata()
    llf.setLengthImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
   
    if labeling!=None:
        data = load_volume(labeling).get_fdata()
        llf.setParcellationImage(nighresjava.JArray('int')(
                                (data.flatten('F')).astype(int).tolist()))

    # set algorithm parameters
    llf.setClusters(clusters)
    
    llf.setThicknesses(nighresjava.JArray('float')(thickness))
    llf.setAngles(nighresjava.JArray('float')(angle))
    llf.setSizes(nighresjava.JArray('float')(size))
    
    llf.setSmooth(smooth)
    llf.setScale(scale)
    
    # execute the algorithm
    try:
    	llf.execute()
    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    data = np.reshape(np.array(llf.getLabelImage(),
                                    dtype=np.int32), shape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(data)
    header['cal_max'] = np.nanmax(data)
    label_img = nb.Nifti1Image(data, affine, header)

    # reshape output to what nibabel likes
    data = np.reshape(np.array(llf.getProbaImage(),
                                    dtype=np.float32), shape=dimensions, order='F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(data)
    header['cal_max'] = np.nanmax(data)
    proba_img = nb.Nifti1Image(data, affine, header)

    # reshape output to what nibabel likes
    if save_data:
        save_volume(label_file, label_img)
        save_volume(proba_file, proba_img)
        return {'label': label_file, 'proba': proba_file}
    else:
        return {'label': label_img, 'proba': proba_img}
        
