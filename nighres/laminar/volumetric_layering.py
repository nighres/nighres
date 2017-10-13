import sys
import os
import numpy as np
import nibabel as nb
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir


def volumetric_layering(inner_levelset, outer_levelset,
                        n_layers=4, topology_lut_dir=None,
                        save_data=False, output_dir=None,
                        file_name=None):

    '''Equivolumetric layering of the cortical sheet.

    Parameters
    ----------
    inner_levelset: niimg
        Levelset representation of the inner surface, typically GM/WM surface
    outer_levelset : niimg
        Levelset representation of the outer surface, typically GM/CSF surface
    n_layers : int, optional
        Number of layers to be created (default is 10)
    topology_lut_dir: str, optional
        Path to directory in which topology files are stored (default is stored
        in TOPOLOGY_LUT_DIR)
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

        * depth (niimg): Continuous depth from 0 (inner surface) to 1
          (outer surface) (_layering_depth)
        * layers (niimg): Discrete layers from 1 (bordering inner surface) to
          n_layers (bordering outer surface) (_layering_layers)
        * boundaries (niimg): Levelset representations of boundaries between
          all layers in 4D (_layering_boundaries)

    Notes
    ----------
    Original Java module by Miriam Waehnert, Pierre-Louis Bazin and
    Juliane Dinse. Algorithm details can be found in [1]_

    References
    ----------
    .. [1] Waehnert et al (2014) Anatomically motivated modeling of cortical
       laminae. DOI: 10.1016/j.neuroimage.2013.03.078
    '''

    print('\nVolumetric Layering')

    # check topology lut dir and set default if not given
    topology_lut_dir = _check_topology_lut_dir(topology_lut_dir)

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, inner_levelset)

        depth_file = _fname_4saving(file_name=file_name,
                                    rootfile=inner_levelset,
                                    suffix='layering_depth')

        layer_file = _fname_4saving(file_name=file_name,
                                    rootfile=inner_levelset,
                                    suffix='layering_layers')

        boundary_file = _fname_4saving(file_name=file_name,
                                       rootfile=inner_levelset,
                                       suffix='layering_boundaries')

    # start virutal machine if not already running
    try:
        cbstools.initVM(initialheap='12000m', maxheap='12000m')
    except ValueError:
        pass

    # initate class
    lamination = cbstools.LaminarVolumetricLayering()

    # load the data
    inner_img = load_volume(inner_levelset)
    inner_data = inner_img.get_data()
    hdr = inner_img.get_header()
    aff = inner_img.get_affine()
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = inner_data.shape

    outer_data = load_volume(outer_levelset).get_data()

    # set parameters from input images
    lamination.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    lamination.setResolutions(resolution[0], resolution[1], resolution[2])
    lamination.setInnerDistanceImage(cbstools.JArray('float')(
                                    (inner_data.flatten('F')).astype(float)))
    lamination.setOuterDistanceImage(cbstools.JArray('float')(
                                    (outer_data.flatten('F')).astype(float)))
    lamination.setNumberOfLayers(n_layers)
    lamination.setTopologyLUTdirectory(topology_lut_dir)

    # execute class
    try:
        lamination.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

    # collect data
    depth_data = np.reshape(np.array(lamination.getContinuousDepthMeasurement(),
                                    dtype=np.float32), dimensions, 'F')
    hdr['cal_max'] = np.nanmax(depth_data)
    depth = nb.Nifti1Image(depth_data, aff, hdr)

    layer_data = np.reshape(np.array(lamination.getDiscreteSampledLayers(),
                                     dtype=np.int32), dimensions, 'F')
    hdr['cal_max'] = np.nanmax(layer_data)
    layers = nb.Nifti1Image(layer_data, aff, hdr)

    boundary_len = lamination.getLayerBoundarySurfacesLength()
    boundary_data = np.reshape(np.array(lamination.getLayerBoundarySurfaces(),
                               dtype=np.float32), (dimensions[0],
                               dimensions[1], dimensions[2], boundary_len),
                               'F')
    hdr['cal_min'] = np.nanmin(boundary_data)
    hdr['cal_max'] = np.nanmax(boundary_data)
    boundaries = nb.Nifti1Image(boundary_data, aff, hdr)

    if save_data:
        save_volume(os.path.join(output_dir, depth_file), depth)
        save_volume(os.path.join(output_dir, layer_file), layers)
        save_volume(os.path.join(output_dir, boundary_file), boundaries)

    return {'depth': depth, 'layers': layers, 'boundaries': boundaries}
