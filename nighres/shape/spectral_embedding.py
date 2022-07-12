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
    _check_topology_lut_dir, _check_available_memory


def spectral_embedding(label_image, 
                    contrasts=None,
                    dims=1,
                    scaling=1.0,
                    msize=800,
                    ref="none",
                    save_data=False, 
                    overwrite=False, 
                    output_dir=None,
                    file_name=None):

    """ Spectral embedding
    
    Derive a spectral Laplacian embedding from labelled regions, optionally taking underlying 
    contrasts into account (technique adapted from [1]).

    Parameters
    ----------
    label_image: niimg
        Image of the object(s) of interest
    contrasts: [niimg], optional
        Additional images with relevant intra-regional contrasts, if required
    dims: int
        Number of kept dimensions in the representation (default is 1)
    scaling: float
        Scaling of intra-regional contrast differences to use (default is 1.0)
    msize: int
        Target matrix size for subsampling (default is 800)
    ref: str
        Reference direction to orient directions ('none', 'X', 'Y', 'Z', default is 'none')
    save_data: bool, optional
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

        * result (niimg): Coordinate map (_se-coord)

    Notes
    ----------
    
    References
    ----------
    .. [1] Orasanu, E., Bazin, P.-L., Melbourne, A., Lorenzi, M., Lombaert, H., Robertson, N.J., 
           Kendall, G., Weiskopf, N., Marlow, N., Ourselin, S., 2016. Longitudinal Analysis of the 
           Preterm Cortex Using Multi-modal Spectral Matching, Proceedings of MICCAI 2016.
           https://doi.org/10.1007/978-3-319-46720-7_30

    """

    print("\nSpectral Shape Embedding")

    if save_data:
        output_dir = _output_dir_4saving(output_dir, label_image)

        coord_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=label_image,
                                  suffix='se-coord'))

        if overwrite is False \
            and os.path.isfile(coord_file) :
                output = {'result': coord_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    algorithm = nighresjava.SpectralShapeEmbedding()

    # load images and set dimensions and resolution
    label_image = load_volume(label_image)
    data = label_image.get_fdata()
    affine = label_image.get_affine()
    header = label_image.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = label_image.shape
    dimensions4 = (dimensions[0],dimensions[1],dimensions[2],4)


    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])

    data = load_volume(label_image).get_fdata()
    algorithm.setLabelImage(nighresjava.JArray('int')(
                               (data.flatten('F')).astype(int).tolist()))
    
    if contrasts is not None:
        algorithm.setContrastNumber(len(contrasts))
        for n,contrast in enumerate(contrasts):
            data = load_volume(contrast).get_fdata()
            algorithm.setContrastImageAt(n, nighresjava.JArray('float')(
                                        (data.flatten('F')).astype(float)))
            algorithm.setContrastDevAt(n, scaling)  
        
    algorithm.setMatrixSize(msize)
    algorithm.setReferenceAxis(ref)
    algorithm.setEigenGame(True,0.1,0.1)
    #algorithm.setEigenGame(True,0.05,0.05)

    # execute
    try:
        algorithm.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # Collect output
    coord_data = np.reshape(np.array(
                                    algorithm.getCoordinateImage(),
                                    dtype=np.float32), dimensions4, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = np.nanmin(coord_data)
    header['cal_max'] = np.nanmax(coord_data)
    coord_img = nb.Nifti1Image(coord_data, affine, header)

    if save_data:
        save_volume(coord_file, coord_img)
        
        return {'result': coord_file}
    else:
        return {'result': coord_img}


def spectral_flatmap(label_image, coord_image,
                    dims=2,
                    size=1024,
                    combined=True,
                    projection=False,
                    offset=0,
                    save_data=False, 
                    overwrite=False, 
                    output_dir=None,
                    file_name=None):

    """ Spectral flat map building
    
    Derive a 2D or 3D flatmap from spectral Laplacian coordinates of labelled regions

    Parameters
    ----------
    label_image: niimg
        Image of the object(s) of interest
    coord_image: niimg
        Corresponding map of coordinates
    dims: int
        Number of kept dimensions in the representation (2 or 3, default is 2)
    size: int
        Target image size to generate (default is 1024)
    combined: bool, optional
        Whether to combine maps into a single representation (default is True)
        Note: this requires more than 3 labels to work properly
    projection: bool, optional
        Whether to use a planar projection along first gradient rather than direct gradient space
        representations (default is False)
    offset: int, optional
        Offset to use different combinations of spectral coordinates (default is 0)
    save_data: bool, optional
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

        * result (niimg): Flat map (_sf-map)

    Notes
    ----------
    

    """

    print("\nSpectral Shape Flat Mapping")

    if save_data:
        output_dir = _output_dir_4saving(output_dir, label_image)

        flatmap_file = os.path.join(output_dir, 
                            _fname_4saving(module=__name__,file_name=file_name,
                                  rootfile=label_image,
                                  suffix='sf-map'))

        if overwrite is False \
            and os.path.isfile(flatmap_file) :
                output = {'result': flatmap_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    algorithm = nighresjava.SpectralShapeEmbedding()

    # load images and set dimensions and resolution
    label_image = load_volume(label_image)
    data = label_image.get_fdata()
    affine = label_image.get_affine()
    header = label_image.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = label_image.shape
    dimensions4 = (dimensions[0],dimensions[1],dimensions[2],4)


    algorithm.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    algorithm.setResolutions(resolution[0], resolution[1], resolution[2])

    data = load_volume(label_image).get_fdata()
    algorithm.setLabelImage(nighresjava.JArray('int')(
                               (data.flatten('F')).astype(int).tolist()))
    
    data = load_volume(coord_image).get_fdata()
    algorithm.setCoordinateImage(nighresjava.JArray('float')(
                               (data.flatten('F')).astype(float)))
                
    # execute
    try:
        if projection: algorithm.buildSpectralProjectionMaps(size, combined)
        else: algorithm.buildSpectralMaps(size, combined, offset)

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # Collect output
    if combined:
        flatdim = (size,size)
    else:       
        flatdim = (size,size,algorithm.getLabelNumber()-1)
    
    flatmap_data = np.reshape(np.array(
                                    algorithm.getFlatMapImage(),
                                    dtype=np.int32), flatdim, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    flatmap_img = nb.Nifti1Image(flatmap_data, None, None)

    if save_data:
        save_volume(flatmap_file, flatmap_img)
        
        return {'result': flatmap_file}
    else:
        return {'result': flatmap_img}


