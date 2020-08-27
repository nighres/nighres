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
                    _check_topology_lut_dir

# convenience labels
X=0
Y=1
Z=2
T=3

def simple_align(source_image, target_image,
                    copy_header=False,
                    align_center=False, 
                    rescale=False,
                    data_type='intensity',
                    ignore_affine=False, ignore_header=False,
                    save_data=False, overwrite=False, output_dir=None,
                    file_name=None):
    """ Simple alignment routines

    Simple routines to align image headers (image data is unchanged)

    Parameters
    ----------
    source_image: niimg
        Image to align
    target_image: niimg
        Reference image to match
    copy_header: bool
        To copy the target header to the source (default is False)
    align_center: bool
        To align the source center of mass to the target (default is False)
    rescale: bool
        To rescale the source to the volume of the target (default is False)
    data_type: {'intensity','nonzero','boundingbox'}
        The type of datato consider for alignment (default is 'intensity')
    ignore_affine: bool
        Ignore the affine matrix information extracted from the image header
        (default is False)
    ignore_header: bool
        Ignore the orientation information and affine matrix information 
        extracted from the image header (default is False)
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

        * result (niimg): Aligne source image (_al-img)

    Notes
    ----------
    This uses the ANTs/ITK conventions with regard to Nifti headers, for better
    or for worse. Note that Nibabel conventions, of course, are different.
    
    """

    print('\nSimple align')

    # make sure that saving related parameters are correct
    output_dir = _output_dir_4saving(output_dir, source_image) # needed for intermediate results
    if save_data:
        result_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=source_image,
                                   suffix='al-img'))

        if overwrite is False \
            and os.path.isfile(result_file) :
            
            print("skip computation (use existing results)")
            output = {'result': result_file}
            return output


    # load and get dimensions and resolution from input images
    source = load_volume(source_image)
    nsx = source.header.get_data_shape()[X]
    nsy = source.header.get_data_shape()[Y]
    nsz = source.header.get_data_shape()[Z]
    rsx = source.header.get_zooms()[X]
    rsy = source.header.get_zooms()[Y]
    rsz = source.header.get_zooms()[Z]

    target = load_volume(target_image)
    ntx = target.header.get_data_shape()[X]
    nty = target.header.get_data_shape()[Y]
    ntz = target.header.get_data_shape()[Z]
    rtx = target.header.get_zooms()[X]
    rty = target.header.get_zooms()[Y]
    rtz = target.header.get_zooms()[Z]

    # in case the affine transformations are not to be trusted: make them equal
    if ignore_affine or ignore_header:
        # create generic affine aligned with the orientation for the source
        mx = np.argmax(np.abs(source.affine[0][0:3]))
        my = np.argmax(np.abs(source.affine[1][0:3]))
        mz = np.argmax(np.abs(source.affine[2][0:3]))
        new_affine = np.zeros((4,4))
        if ignore_header:
            new_affine[0][0] = rsx
            new_affine[1][1] = rsy
            new_affine[2][2] = rsz
            new_affine[0][3] = -rsx*nsx/2.0
            new_affine[1][3] = -rsy*nsy/2.0
            new_affine[2][3] = -rsz*nsz/2.0
        else:
            new_affine[0][mx] = rsx*np.sign(source.affine[0][mx])
            new_affine[1][my] = rsy*np.sign(source.affine[1][my])
            new_affine[2][mz] = rsz*np.sign(source.affine[2][mz])
            if (np.sign(source.affine[0][mx])<0): 
                new_affine[0][3] = rsx*nsx/2.0
            else:
                new_affine[0][3] = -rsx*nsx/2.0
                
            if (np.sign(source.affine[1][my])<0): 
                new_affine[1][3] = rsy*nsy/2.0
            else:
                new_affine[1][3] = -rsy*nsy/2.0
                
            if (np.sign(source.affine[2][mz])<0): 
                new_affine[2][3] = rsz*nsz/2.0
            else:
                new_affine[2][3] = -rsz*nsz/2.0
        #if (np.sign(source.affine[0][mx])<0): new_affine[mx][3] = rsx*nsx
        #if (np.sign(source.affine[1][my])<0): new_affine[my][3] = rsy*nsy
        #if (np.sign(source.affine[2][mz])<0): new_affine[mz][3] = rsz*nsz
        #new_affine[0][3] = nsx/2.0
        #new_affine[1][3] = nsy/2.0
        #new_affine[2][3] = nsz/2.0
        new_affine[3][3] = 1.0
        
        source = nb.Nifti1Image(source.get_data(), new_affine, source.header)
        source.update_header()
        
        # create generic affine aligned with the orientation for the target
        mx = np.argmax(np.abs(target.affine[0][0:3]))
        my = np.argmax(np.abs(target.affine[1][0:3]))
        mz = np.argmax(np.abs(target.affine[2][0:3]))
        new_affine = np.zeros((4,4))
        if ignore_header:
            new_affine[0][0] = rtx
            new_affine[1][1] = rty
            new_affine[2][2] = rtz
            new_affine[0][3] = -rtx*ntx/2.0
            new_affine[1][3] = -rty*nty/2.0
            new_affine[2][3] = -rtz*ntz/2.0
        else:
            new_affine[0][mx] = rtx*np.sign(target.affine[0][mx])
            new_affine[1][my] = rty*np.sign(target.affine[1][my])
            new_affine[2][mz] = rtz*np.sign(target.affine[2][mz])
            if (np.sign(target.affine[0][mx])<0): 
                new_affine[0][3] = rtx*ntx/2.0
            else:
                new_affine[0][3] = -rtx*ntx/2.0
                
            if (np.sign(target.affine[1][my])<0): 
                new_affine[1][3] = rty*nty/2.0
            else:
                new_affine[1][3] = -rty*nty/2.0
                
            if (np.sign(target.affine[2][mz])<0): 
                new_affine[2][3] = rtz*ntz/2.0
            else:
                new_affine[2][3] = -rtz*ntz/2.0
        #if (np.sign(target.affine[0][mx])<0): new_affine[mx][3] = rtx*ntx
        #if (np.sign(target.affine[1][my])<0): new_affine[my][3] = rty*nty
        #if (np.sign(target.affine[2][mz])<0): new_affine[mz][3] = rtz*ntz
        #new_affine[0][3] = ntx/2.0
        #new_affine[1][3] = nty/2.0
        #new_affine[2][3] = ntz/2.0
        new_affine[3][3] = 1.0
        
        target = nb.Nifti1Image(target.get_data(), new_affine, target.header)
        target.update_header()
    
    # compute the various options
    if copy_header:
       result = nb.Nifti1Image(source.get_data(), target.affine, target.header)
       result.update_header()
         
    else:
        if align_center:
            # compute source center etc etc
            src_center = np.zeros(4)
            trg_center = np.zeros(4)
            
            if data_type == 'intensity':
                 for x in range(nsx):
                    for y in range(nsy):
                        for z in range(nsz):
                            src_center[X] += x*source.get_data()[x,y,z]
                            src_center[Y] += y*source.get_data()[x,y,z]
                            src_center[Z] += z*source.get_data()[x,y,z]
                            src_center[T] += source.get_data()[x,y,z]
                 for x in range(ntx):
                    for y in range(nty):
                        for z in range(ntz):
                            trg_center[X] += x*target.get_data()[x,y,z]
                            trg_center[Y] += y*target.get_data()[x,y,z]
                            trg_center[Z] += z*target.get_data()[x,y,z]
                            trg_center[T] += target.get_data()[x,y,z]
            elif data_type == 'nonzero':
                 for x in range(nsx):
                    for y in range(nsy):
                        for z in range(nsz):
                            if source.get_data()[x,y,z]>0:
                                src_center[X] += x
                                src_center[Y] += y
                                src_center[Z] += z
                                src_center[T] += 1
                 for x in range(ntx):
                    for y in range(nty):
                        for z in range(ntz):
                            if target.get_data()[x,y,z]>0:
                                trg_center[X] += x
                                trg_center[Y] += y
                                trg_center[Z] += z
                                trg_center[T] += 1
            elif data_typ == 'boundingbox':
                src_center[X] = nsx/2
                src_center[Y] = nsy/2
                src_center[Z] = nsz/2
                src_center[T] = 1
                trg_center[X] = nsx/2
                trg_center[Y] = nsy/2
                trg_center[Z] = nsz/2
                trg_center[T] = 1
            
            src_center[X] /= src_center[T]
            src_center[Y] /= src_center[T]
            src_center[Z] /= src_center[T]
            trg_center[X] /= trg_center[T]
            trg_center[Y] /= trg_center[T]
            trg_center[Z] /= trg_center[T]
            
            source.affine[X,T] = target.affine[X,T] - rsx*src_center[X] \
                                                    + ntx*trg_center[X]
            source.affine[Y,T] = target.affine[Y,T] - rsy*src_center[Y] \
                                                    + nty*trg_center[Y]
            source.affine[Z,T] = target.affine[Z,T] - rsz*src_center[Z] \
                                                    + ntz*trg_center[Z]

        if rescale:
            src_size=0
            trg_size=0
            if data_type == 'intensity':
                 for x in range(nsx):
                    for y in range(nsy):
                        for z in range(nsz):
                            src_size += source.get_data()[x,y,z]
                 for x in range(ntx):
                    for y in range(nty):
                        for z in range(ntz):
                            trg_size += target.get_data()[x,y,z]
            elif data_type == 'nonzero':
                 for x in range(nsx):
                    for y in range(nsy):
                        for z in range(nsz):
                            if source.get_data()[x,y,z]>0:
                                src_size += 1
                 for x in range(ntx):
                    for y in range(nty):
                        for z in range(ntz):
                            if target.get_data()[x,y,z]>0:
                                trg_size += 1
            elif data_typ == 'boundingbox':
                src_size = nsx*nsy*nsz
                trg_size = ntx*nty*ntz
            
            source.affine = trg_size/src_size*source.affine
            source.affine[T,T] = 1
            
        result = nb.Nifti1Image(source.get_data(), source.affine, source.header)
        result.update_header()

    if save_data:
        save_volume(result_file, result)
        outputs = {'result': result_file}
    else:
        outputs = {'result': result}

    return outputs
