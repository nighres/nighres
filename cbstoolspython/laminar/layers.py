import os
import numpy as np
import nibabel as nb
import cbstools
from ..io import load_volume, save_volume


def layering(gwb_levelset, cgb_levelset, n_layers=10, lut_dir='lookuptables/',
             save_data=True, base_name=None):

    '''
    Equivolumetric layering of the cortical sheet.

    Waehnert et al. (2014). Anatomically motivated modeling of cortical
    laminae. http://doi.org/10.1016/j.neuroimage.2013.03.078

        Parameters
        -----------
        gwb_levelset : Levelset representation of the GM/WM surface. Can be
            created from tissue segmentation with the "create_levelsets"
            function. Can be path to a Nifti file or Nibabel image object.
        cgb_levelset : Levelset representation of the CSF/GM surface. Can be
            created from tissue segmentation with the "create_levelsets"
            function. Can be path to a Nifti file or Nibabel image object.
        n_layers : int, number of layers to be created.
        lut_dir : Path to directory with lookup tables. Default is to search it
            within this directory.
        save_data : Whether the output layer image should be saved
            (default is 'True').
        base_name : If save_data is set to True, this parameter can be used to
            specify where the output should be saved. Thus can be the path to a
            directory or a full filename. The suffixes 'depth', 'layers' and
            'boundaries' will be added to the respective outputs. If None
            (default), the output will be saved to the current directory.

        Returns
        -------
        Three Nibabel image objects :
            Continuous depth from 0(WM) to 1(CSF)
            Discrete layers from 1(bordering WM) to n_layers(bordering CSF)
            Levelset representations of boundaries between layers (4D)
    '''

    # load the data as well as filenames and headers for saving later
    gwb_img = load_volume(gwb_levelset)
    gwb_data = gwb_img.get_data()
    hdr = gwb_img.get_header()
    aff = gwb_img.get_affine()

    cgb_data = load_volume(cgb_levelset).get_data()

    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass

    lamination = cbstools.LaminarVolumetricLayering()
    lamination.setDimensions(gwb_data.shape[0], gwb_data.shape[1], gwb_data.shape[2])
    zooms = [x.item() for x in hdr.get_zooms()]
    lamination.setResolutions(zooms[0], zooms[1], zooms[2])

    lamination.setInnerDistanceImage(cbstools.JArray('float')((gwb_data.flatten('F')).astype(float)))
    lamination.setOuterDistanceImage(cbstools.JArray('float')((cgb_data.flatten('F')).astype(float)))
    lamination.setNumberOfLayers(n_layers)
    lamination.setTopologyLUTdirectory(lut_dir)
    lamination.execute()

    depth_data=np.reshape(np.array(lamination.getContinuousDepthMeasurement(), dtype=np.float32),gwb_data.shape,'F')
    layer_data=np.reshape(np.array(lamination.getDiscreteSampledLayers(), dtype=np.uint32),gwb_data.shape,'F')

    boundary_len = lamination.getLayerBoundarySurfacesLength()
    boundary_data=np.reshape(np.array(lamination.getLayerBoundarySurfaces(), dtype=np.float32),
                             (gwb_data.shape[0], gwb_data.shape[1],
                              gwb_data.shape[2],
                              boundary_len), 'F')

    depth_img = nb.Nifti1Image(depth_data, aff, hdr)
    layer_img = nb.Nifti1Image(layer_data, aff, hdr)
    boundary_img = nb.Nifti1Image(boundary_data, aff, hdr)

    if save_data:
        if base_name:
            base_name += '_'
        else:
            if not isinstance(gwb_levelset, basestring):
                base_name = os.getcwd() + '/'
                print "saving to %s" % base_name
            else:
                dir_name = os.path.dirname(gwb_levelset)
                base_name = os.path.basename(gwb_levelset)
                base_name = os.path.join(dir_name,
                                         base_name[:base_name.find('.')]) + '_'
                print "saving to %s" % base_name

        save_volume(base_name + 'depth.nii.gz', depth_img)
        save_volume(base_name + 'layers.nii.gz', layer_img)
        save_volume(base_name + 'boundaries.nii.gz', boundary_img)

    return depth_img, layer_img, boundary_img
