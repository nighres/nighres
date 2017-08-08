import os
import numpy as np
import nibabel as nb
import cbstools
from ..io import load_volume, save_volume
from ..global_settings import TOPOLOGY_LUT_DIR


def volumetric_layering(inner_levelset, outer_levelset,
                        n_layers=10, topology_lut_dir=None,
                        save_data=False, output_dir=None,
                        file_name=None, file_extension=None):

    '''Equivolumetric layering of the cortical sheet.

    Parameters
    ----------
    inner_levelset: TODO:type
        Levelset representation of the inner surface, typically GM/WM surface
    outer_levelset : TODO:type
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
        Desired base name for output files (suffixes will be added)
    file_extension: str, optional
        Desired extension for output files (determines file type)

    Returns
    ----------
    outputs: dict
        Dictionary collecting outputs under the following keys
        - 'depth': Continuous depth from 0 (inner surface) to 1 (outer surface)
                   (layering_depth)
        - 'layers': Discrete layers from 1 (bordering inner surface) to
                    n_layers (bordering outer surface) (_layering_layers)
        - 'boundaries': Levelset representations of boundaries between
                        all layers (4D image) (_layering_boundaries)
        (suffix of output files if save_data is set to True)

    Notes
    ----------
    Original Java module by Miriam Waehnert, Pierre-Louis Bazin and
    Juliane Dinse. Algorithm details can be found in [1].

    References
    ----------
    [1] Waehnert et al (2014). Anatomically motivated modeling of cortical
    laminae. DOI: 10.1016/j.neuroimage.2013.03.078
    '''

    # set default topology lut dir if not given
    if topology_lut_dir is None:
        topology_lut_dir = TOPOLOGY_LUT_DIR
    else:
        # check if dir exists
        if not os.path.isdir(topology_lut_dir):
            raise ValueError('The topology_lut_dir you have specified ({0}) '
                             'does not exist'.format(topology_lut_dir))
        # if we don't end in a path sep, we need to make sure that we add it
        if not(topology_lut_dir[-1] == os.path.sep):
            topology_lut_dir += os.path.sep

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, inner_levelset)

        depth_file = _fname_4saving(rootfile=inner_image,
                                    suffix='layering_depth',
                                    base_name=file_name,
                                    extension=file_extension)

        layer_file = _fname_4saving(rootfile=inner_image,
                                    suffix='layering_layers',
                                    base_name=file_name,
                                    extension=file_extension)

        boundary_file = _fname_4saving(rootfile=inner_image,
                                       suffix='layering_boundaries',
                                       base_name=file_name,
                                       extension=file_extension)

    # start virutal machine if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
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
        print("Executing volumetric layering")
        lamination.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print sys.exc_info()[0]
        raise
        return

    # collect data
    depth_data = np.reshape(np.array(
                                    lamination.getContinuousDepthMeasurement(),
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
    hdr['cal_max'] = np.nanmax(boundary_data)
    boundaries = nb.Nifti1Image(boundary_data, aff, hdr)

    if save_data:
        save_volume(os.path.join(output_dir, depth_file), depths)
        save_volume(os.path.join(output_dir, layer_file), layers)
        save_volume(os.path.join(output_dir, boundary_file), boundaries)

    return {'depth': depth, 'layers': layers, 'boundaries': boundaries}
