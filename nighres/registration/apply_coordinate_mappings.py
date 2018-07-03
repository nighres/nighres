import os
import numpy as np
import nibabel as nb
import sys
import cbstools
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir


def apply_coordinate_mappings(image, mapping1, 
                        mapping2=None, mapping3=None, mapping4=None,
                        interpolation="nearest", padding="closest",
                        save_data=False, overwrite=False, output_dir=None,
                        file_name=None):

    '''Apply a coordinate mapping (or a succession of coordinate mappings) to
        a 3D or 4D image.

    Parameters
    ----------
    image: niimg
        Image to deform
    mapping1 : niimg
        First coordinate mapping to apply
    mapping2 : niimg, optional
        Second coordinate mapping to apply
    mapping3 : niimg, optional
        Third coordinate mapping to apply
    mapping4 : niimg, optional
        Fourth coordinate mapping to apply
    interpolation: {'nearest', 'linear', 'wsinc'}
        Interpolation method (default is 'nearest')
    padding: {'closest', 'zero', 'max'}
        Image padding method (default is 'closest')
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
    niimg
       Result image (output file suffix _def_img)
       
    Notes
    ----------
    Original Java module by Pierre-Louis Bazin
    
    '''

    print('\nApply coordinate mappings')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        deformed_file = os.path.join(output_dir, 
                        _fname_4saving(file_name=file_name,
                                    rootfile=image,
                                    suffix='def_img'))
        if overwrite is False \
            and os.path.isfile(deformed_file) :
            
            print("skip computation (use existing results)")
            output = load_volume(deformed_file)
            return output

    # start virutal machine if not already running
    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass

    # initate class
    applydef = cbstools.RegistrationApplyDeformations()

    # load the data
    img = load_volume(image)
    data = img.get_data()
    hdr = img.get_header()
    aff = img.get_affine()
    imgres = [x.item() for x in hdr.get_zooms()]
    imgdim = data.shape

    # set parameters from input images
    applydef.setImageDimensions(imgdim[0], imgdim[1], imgdim[2])
    applydef.setImageResolutions(imgres[0], imgres[1], imgres[2])
    
    applydef.setImageToDeform(cbstools.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
    def1data = load_volume(mapping1).get_data()
    applydef.setDeformationMapping1(cbstools.JArray('float')(
                                    (def1data.flatten('F')).astype(float)))
    applydef.setDeformation1Dimensions(def1data.shape[0],def1data.shape[1],def1data.shape[2])
    applydef.setDeformationType1("mapping(voxels)")
    
    if not (mapping2==None):
        def2data = load_volume(mapping2).get_data()
        applydef.setDeformationMapping2(cbstools.JArray('float')(
                                        (def2data.flatten('F')).astype(float)))
        applydef.setDeformation2Dimensions(def2data.shape[0],def2data.shape[1],def2data.shape[2])
        applydef.setDeformationType2("mapping(voxels)")
        
        if not (mapping3==None):
            def3data = load_volume(mapping3).get_data()
            applydef.setDeformationMapping3(cbstools.JArray('float')(
                                            (def3data.flatten('F')).astype(float)))
            applydef.setDeformation3Dimensions(def3data.shape[0],def3data.shape[1],def3data.shape[2])
            applydef.setDeformationType3("mapping(voxels)")
        
            if not (mapping4==None):
                def4data = load_volume(mapping4).get_data()
                applydef.setDeformationMapping4(cbstools.JArray('float')(
                                                (def4data.flatten('F')).astype(float)))
                applydef.setDeformation4Dimensions(def4data.shape[0],def4data.shape[1],def4data.shape[2])
                applydef.setDeformationType4("mapping(voxels)")
        
    applydef.setInterpolationType(interpolation)
    applydef.setImagePadding(padding)

    # execute class
    try:
        applydef.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # collect data
    deformed_data = np.reshape(np.array(
                                applydef.getDeformedImage(),
                                dtype=np.float32), imgdim, 'F')
    hdr['cal_min'] = np.nanmin(deformed_data)
    hdr['cal_max'] = np.nanmax(deformed_data)
    deformed = nb.Nifti1Image(deformed_data, aff, hdr)

    if save_data:
        save_volume(os.path.join(output_dir, deformed_file), deformed)

    return deformed
