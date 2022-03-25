import numpy
import nibabel
import os
import sys
import nighresjava
from ..io import load_volume, save_volume, load_mesh
from ..utils import _output_dir_4saving, _fname_4saving, \
                    _check_topology_lut_dir, _check_available_memory


def mesh_data_to_volume(surface_mesh, reference_image, 
                      combine='mean', distance_mm=1.0, center=False,
                      rescale=None,
                      save_data=False, overwrite=False, output_dir=None,
                      file_name=None):
    """ Mesh data to volume

    Projects mesh data onto voxel space and propagates the values into the neighboring voxels

    Parameters
    ----------
    surface_mesh: mesh
        Mesh model of the surface
    reference_image: niimg
        Image of the dimensions and resolutions corresponding to the mesh
    combine: {'min','mean','max'}, optional
        Propagate using the mean (default), max or min data from neighboring voxels
    distance_mm: float, optional 
        Distance for the propagation (note: this algorithm will be slow for 
        large distances)
    center: bool
        Use image center as origin (default is False)
    rescale: float or array
        Rescale the data by a given factor (or factors for each axis, default is None)
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

        * result (niimg): The propagated mesh data image

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.

    """

    print('\nMesh data to volume')

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, surface_mesh)

        out_file = os.path.join(output_dir, 
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=surface_mesh,
                                   suffix='mdtv-img', ext="nii.gz"))

        if overwrite is False \
            and os.path.isfile(out_file) :
                print("skip computation (use existing results)")
                output = {'result': out_file}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create instance
    propag = nighresjava.IntensityPropagate()

    # set parameters
    
    # load the data
    mesh = load_mesh(surface_mesh)

    # load image and use it to set dimensions and resolution
    img = load_volume(reference_image)
    affine = img.affine
    header = img.header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = img.header.get_data_shape()

    # recenter to middle of the image if required
    cx = 0.0
    cy = 0.0
    cz = 0.0    
    if center is not None:
        if isinstance(center, list):
            cx = center[0]
            cy = center[1]
            cz = center[2]
        elif center:
            cx = dimensions[0]/2.0
            cy = dimensions[1]/2.0
            cz = dimensions[2]/2.0       

    #rescale if required
    rx = 1.0
    ry = 1.0
    rz = 1.0
    if rescale is not None:
        if isinstance(rescale, list):
            rx = rescale[0]
            ry = rescale[1]
            rz = rescale[2]
        else:
            rx = rescale
            ry = rescale
            rz = rescale

    # paste the mesh values onto the reference image space
    data = numpy.zeros(dimensions)
    for idx,val in enumerate(mesh['points']):
        x = int(round(mesh['points'][idx][0]*rx+cx))
        y = int(round(mesh['points'][idx][1]*ry+cy))
        z = int(round(mesh['points'][idx][2]*rz+cz))
        
        if (x>=0 and x<dimensions[0] and y>=0 and y<dimensions[1] and z>=0 and z<dimensions[2]):
            data[x,y,z] = mesh['data'][idx][0]

    propag.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    propag.setResolutions(resolution[0], resolution[1], resolution[2])
        
    propag.setInputImage(nighresjava.JArray('float')(
                                    (data.flatten('F')).astype(float)))
    
       
    # set algorithm parameters
    propag.setCombinationMethod(combine)
    propag.setPropagationDistance(distance_mm)
    propag.setTargetVoxels('zero')
    propag.setPropogationScalingFactor(1.0)
    
    # execute the algorithm
    try:
        propag.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # reshape output to what nibabel likes
    propag_data = numpy.reshape(numpy.array(propag.getResultImage(),
                                    dtype=numpy.float32), dimensions, 'F')

    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
    header['cal_min'] = numpy.nanmin(propag_data)
    header['cal_max'] = numpy.nanmax(propag_data)
    out = nibabel.Nifti1Image(propag_data, affine, header)

    if save_data:
        save_volume(out_file, out)
        return {'result': out_file}
    else:
        return {'result': out}
