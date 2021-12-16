import os
import numpy as np
import nibabel as nb
import sys
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def apply_coordinate_mappings(image, mapping1,
                        mapping2=None, mapping3=None, mapping4=None,
                        interpolation="nearest", padding="closest",
                        save_data=False, overwrite=False, output_dir=None,
                        file_name=None):

    '''Apply a coordinate mapping (or a succession of coordinate mappings) to a 3D or 4D image.

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
    interpolation: {'nearest', 'linear'}
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
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * result (niimg): Result image (_def-img)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin

    '''

    print('\nApply coordinate mappings')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        deformed_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,
                                    file_name=file_name,
                                    rootfile=image,
                                    suffix='def-img'))
        if overwrite is False \
            and os.path.isfile(deformed_file) :

            print("skip computation (use existing results)")
            output = {'result': deformed_file}
            return output

    # start virutal machine if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initate class
    applydef = nighresjava.RegistrationApplyDeformations()

    # load the data
    img = load_volume(image)
    data = img.get_data()
    hdr = img.header
    aff = img.affine
    imgres = [x.item() for x in hdr.get_zooms()]
    imgdim = data.shape

    # set parameters from input images
    if len(imgdim)==4:
        applydef.setImageDimensions(imgdim[0], imgdim[1], imgdim[2], imgdim[3])
    else:
        applydef.setImageDimensions(imgdim[0], imgdim[1], imgdim[2])
    applydef.setImageResolutions(imgres[0], imgres[1], imgres[2])

    applydef.setImageToDeform(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

    def1 = load_volume(mapping1)
    def1data = def1.get_data()
    aff = def1.affine
    hdr = def1.header
    trgdim = def1data.shape
    applydef.setDeformationMapping1(nighresjava.JArray('float')(
                                    (def1data.flatten('F')).astype(float)))
    applydef.setDeformation1Dimensions(def1data.shape[0],
                                        def1data.shape[1],def1data.shape[2])
    applydef.setDeformationType1("mapping(voxels)")

    if not (mapping2==None):
        def2 = load_volume(mapping2)
        def2data = def2.get_data()
        aff = def2.affine
        hdr = def2.header
        trgdim = def2data.shape
        applydef.setDeformationMapping2(nighresjava.JArray('float')(
                                        (def2data.flatten('F')).astype(float)))
        applydef.setDeformation2Dimensions(def2data.shape[0],
                                        def2data.shape[1],def2data.shape[2])
        applydef.setDeformationType2("mapping(voxels)")

        if not (mapping3==None):
            def3 = load_volume(mapping3)
            def3data = def3.get_data()
            aff = def3.affine
            hdr = def3.header
            trgdim = def3data.shape
            applydef.setDeformationMapping3(nighresjava.JArray('float')(
                                            (def3data.flatten('F')).astype(float)))
            applydef.setDeformation3Dimensions(def3data.shape[0],
                                            def3data.shape[1],def3data.shape[2])
            applydef.setDeformationType3("mapping(voxels)")

            if not (mapping4==None):
                def4 = load_volume(mapping4)
                def4data = def4.get_data()
                aff = def4.affine
                hdr = def4.header
                trgdim = def4data.shape
                applydef.setDeformationMapping4(nighresjava.JArray('float')(
                                        (def4data.flatten('F')).astype(float)))
                applydef.setDeformation4Dimensions(def4data.shape[0],
                                            def4data.shape[1],def4data.shape[2])
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
    if len(imgdim)==4:
        trgdim = [trgdim[0],trgdim[1],trgdim[2],imgdim[3]]
    else:
        trgdim = [trgdim[0],trgdim[1],trgdim[2]]
    deformed_data = np.reshape(np.array(
                                applydef.getDeformedImage(),
                                dtype=np.float32), trgdim, 'F')
    hdr['cal_min'] = np.nanmin(deformed_data)
    hdr['cal_max'] = np.nanmax(deformed_data)
    deformed = nb.Nifti1Image(deformed_data, aff, hdr)

    if save_data:
        save_volume(deformed_file, deformed)
        return {'result': deformed_file}
    else:
        return {'result': deformed}

def apply_coordinate_mappings_2d(image, mapping1,
                        mapping2=None, mapping3=None, mapping4=None,
                        interpolation="nearest", padding="closest",
                        save_data=False, overwrite=False, output_dir=None,
                        file_name=None):

    '''Apply a 2D coordinate mapping (or a succession of coordinate mappings) to a 2D or 3D image.

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
    interpolation: {'nearest', 'linear'}
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
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * result (niimg): Result image (_def-img)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin

    '''

    print('\nApply coordinate mappings (2D)')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        deformed_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                    rootfile=image,
                                    suffix='def-img'))
        if overwrite is False \
            and os.path.isfile(deformed_file) :

            print("skip computation (use existing results)")
            output = {'result': load_volume(deformed_file)}
            return output

    # start virutal machine if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initate class
    applydef = nighresjava.RegistrationApplyDeformations2D()

    # load the data
    img = load_volume(image)
    data = img.get_data()
    hdr = img.header
    aff = img.affine
    imgres = [x.item() for x in hdr.get_zooms()]
    imgdim = data.shape

    # set parameters from input images
    if len(imgdim)==3:
        applydef.setImageDimensions(imgdim[0], imgdim[1], imgdim[2])
    else:
        applydef.setImageDimensions(imgdim[0], imgdim[1])
    applydef.setImageResolutions(imgres[0], imgres[1])

    applydef.setImageToDeform(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

    def1 = load_volume(mapping1)
    def1data = def1.get_data()
    aff = def1.affine
    hdr = def1.header
    trgdim = def1data.shape
    applydef.setDeformationMapping1(nighresjava.JArray('float')(
                                    (def1data.flatten('F')).astype(float)))
    applydef.setDeformation1Dimensions(def1data.shape[0],
                                        def1data.shape[1])
    applydef.setDeformationType1("mapping(voxels)")

    if not (mapping2==None):
        def2 = load_volume(mapping2)
        def2data = def2.get_data()
        aff = def2.affine
        hdr = def2.header
        trgdim = def2data.shape
        applydef.setDeformationMapping2(nighresjava.JArray('float')(
                                        (def2data.flatten('F')).astype(float)))
        applydef.setDeformation2Dimensions(def2data.shape[0],
                                        def2data.shape[1])
        applydef.setDeformationType2("mapping(voxels)")

        if not (mapping3==None):
            def3 = load_volume(mapping3)
            def3data = def3.get_data()
            aff = def3.affine
            hdr = def3.header
            trgdim = def3data.shape
            applydef.setDeformationMapping3(nighresjava.JArray('float')(
                                            (def3data.flatten('F')).astype(float)))
            applydef.setDeformation3Dimensions(def3data.shape[0],
                                            def3data.shape[1])
            applydef.setDeformationType3("mapping(voxels)")

            if not (mapping4==None):
                def4 = load_volume(mapping4)
                def4data = def4.get_data()
                aff = def4.affine
                hdr = def4.header
                trgdim = def4data.shape
                applydef.setDeformationMapping4(nighresjava.JArray('float')(
                                        (def4data.flatten('F')).astype(float)))
                applydef.setDeformation4Dimensions(def4data.shape[0],
                                            def4data.shape[1])
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
    if len(imgdim)==3:
        trgdim = [trgdim[0],trgdim[1],imgdim[2]]
    else:
        trgdim = [trgdim[0],trgdim[1]]
    deformed_data = np.reshape(np.array(
                                applydef.getDeformedImage(),
                                dtype=np.float32), trgdim, 'F')
    hdr['cal_min'] = np.nanmin(deformed_data)
    hdr['cal_max'] = np.nanmax(deformed_data)
    deformed = nb.Nifti1Image(deformed_data, aff, hdr)

    if save_data:
        save_volume(deformed_file, deformed)
        return {'result': deformed_file}
    else:
        return {'result': deformed}


def apply_angular_coordinate_mappings_2d(image, mapping1,
                        mapping2=None, mapping3=None, mapping4=None,
                        interpolation="nearest", padding="closest", unit="rad",
                        save_data=False, overwrite=False, output_dir=None,
                        file_name=None):

    '''Apply an 2D coordinate mapping (or a succession of coordinate mappings) to a 2D image
    with angular information. The angle is updated to reflect local rotations.

    Parameters
    ----------
    image: niimg
        Image to deform, in radians
    mapping1 : niimg
        First coordinate mapping to apply
    mapping2 : niimg, optional
        Second coordinate mapping to apply
    mapping3 : niimg, optional
        Third coordinate mapping to apply
    mapping4 : niimg, optional
        Fourth coordinate mapping to apply
    interpolation: {'nearest', 'linear'}
        Interpolation method (default is 'nearest')
    padding: {'closest', 'zero', 'max'}
        Image padding method (default is 'closest')
    unit: {'deg', 'rad'}
        Angular unit (default is 'rad')
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

        * result (niimg): Result image (_def-img)

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin

    '''

    print('\nApply coordinate mappings (2D)')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, image)

        deformed_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                    rootfile=image,
                                    suffix='def-img'))
        if overwrite is False \
            and os.path.isfile(deformed_file) :

            print("skip computation (use existing results)")
            output = {'result': load_volume(deformed_file)}
            return output

    # start virutal machine if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass

    # initate class
    applydef = nighresjava.RegistrationApplyAngularDeformations2D()

    # load the data
    img = load_volume(image)
    data = img.get_data()
    hdr = img.header
    aff = img.affine
    imgres = [x.item() for x in hdr.get_zooms()]
    imgdim = data.shape

    # convert units if needed
    if unit=='deg':
        data = data/180.0*np.pi

    # set parameters from input images
    if len(imgdim)==3:
        applydef.setImageDimensions(imgdim[0], imgdim[1], imgdim[2])
    else:
        applydef.setImageDimensions(imgdim[0], imgdim[1])
    applydef.setImageResolutions(imgres[0], imgres[1])

    applydef.setImageToDeform(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))

    def1 = load_volume(mapping1)
    def1data = def1.get_data()
    aff = def1.affine
    hdr = def1.header
    trgdim = def1data.shape
    applydef.setDeformationMapping1(nighresjava.JArray('float')(
                                    (def1data.flatten('F')).astype(float)))
    applydef.setDeformation1Dimensions(def1data.shape[0],
                                        def1data.shape[1])
    applydef.setDeformationType1("mapping(voxels)")

    if not (mapping2==None):
        def2 = load_volume(mapping2)
        def2data = def2.get_data()
        aff = def2.affine
        hdr = def2.header
        trgdim = def2data.shape
        applydef.setDeformationMapping2(nighresjava.JArray('float')(
                                        (def2data.flatten('F')).astype(float)))
        applydef.setDeformation2Dimensions(def2data.shape[0],
                                        def2data.shape[1])
        applydef.setDeformationType2("mapping(voxels)")

        if not (mapping3==None):
            def3 = load_volume(mapping3)
            def3data = def3.get_data()
            aff = def3.affine
            hdr = def3.header
            trgdim = def3data.shape
            applydef.setDeformationMapping3(nighresjava.JArray('float')(
                                            (def3data.flatten('F')).astype(float)))
            applydef.setDeformation3Dimensions(def3data.shape[0],
                                            def3data.shape[1])
            applydef.setDeformationType3("mapping(voxels)")

            if not (mapping4==None):
                def4 = load_volume(mapping4)
                def4data = def4.get_data()
                aff = def4.affine
                hdr = def4.header
                trgdim = def4data.shape
                applydef.setDeformationMapping4(nighresjava.JArray('float')(
                                        (def4data.flatten('F')).astype(float)))
                applydef.setDeformation4Dimensions(def4data.shape[0],
                                            def4data.shape[1])
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
    if len(imgdim)==3:
        trgdim = [trgdim[0],trgdim[1],imgdim[2]]
    else:
        trgdim = [trgdim[0],trgdim[1]]
    deformed_data = np.reshape(np.array(
                                applydef.getDeformedImage(),
                                dtype=np.float32), trgdim, 'F')
    
    if unit=='deg':
        deformed_data = deformed_data/np.pi*180.0
    
    hdr['cal_min'] = np.nanmin(deformed_data)
    hdr['cal_max'] = np.nanmax(deformed_data)
    deformed = nb.Nifti1Image(deformed_data, aff, hdr)

    if save_data:
        save_volume(deformed_file, deformed)
        return {'result': deformed_file}
    else:
        return {'result': deformed}
