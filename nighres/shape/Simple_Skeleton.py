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
    _check_topology_lut_dir, _check_atlas_file, _check_available_memory


def Simple_Skeleton(input_image,
		   shape_image_type = 'signed_distance',
                   boundary_threshold = 0.0,
                   skeleton_threshold = 2.0,
		   Topology_LUT_directory = None,
                   save_data=False, 
                   overwrite=False, 
                   output_dir=None,
                   file_name=None):

    """ Simple Skeleton
    
    Create a skeleton for a levelset surface or a probability map (loosely adapted from Bouix et al., 2006)


    Parameters
    ----------
    input_image: niimg
        Image containing structure-of-interest
    shape_image_type: str
        shape of the input image: either 'signed_distance' or 'probability_map'.
    boundary_threshold: float
	Boundary threshold (>0: inside, <0: outside)
    skeleton_threshold: float
	Skeleton threshold (>0: inside, <0: outside)
    Topology_LUT_directory:str
         Directory of LUT topology
    save_data: bool, optional
        Save output data to file (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist
    file_name: str, optional
        Desired base name for output files with file extension
        (suffixes will be added)

    Returns
    ----------
   Medial_Surface_Image
   Medial_Curve_Image

    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    if save_data:
        output_dir = _output_dir_4saving(output_dir, input_image)

        MedialSurface_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='medial', )

        Medial_Curve_file = _fname_4saving(file_name=file_name,
                                  rootfile=input_image,
                                  suffix='skel')     

        if overwrite is False \
            and os.path.isfile(MedialSurface_file) \
            and os.path.isfile(Medial_Curve_file) :
                output = {'medial': load_volume(MedialSurface_file),
                          'skel': load_volume(Medial_Curve_file)}
                return output

    # start virtual machine, if not already running
    try:
        mem = _check_available_memory()
        nighresjava.initVM(initialheap=mem['init'], maxheap=mem['max'])
    except ValueError:
        pass
    # create algorithm instance
    skeleton = nighresjava.ShapeSimpleSkeleton()

    # set parameters
    skeleton.setBoundaryThreshold(boundary_threshold)
    skeleton.setSkeletonThreshold(skeleton_threshold)
    skeleton.setTopologyLUTdirectory(Topology_LUT_directory)
    skeleton.setShapeImageType(shape_image_type)


    # load images and set dimensions and resolution
    input_image = load_volume(input_image)
    data = input_image.get_data()
    affine = input_image.get_affine()
    header = input_image.get_header()
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = input_image.shape


    skeleton.setDimensions(dimensions[0], dimensions[1], dimensions[2])
    skeleton.setResolutions(resolution[0], resolution[1], resolution[2])

    data = load_volume(input_image).get_data()
    skeleton.setShapeImage(nighresjava.JArray('float')(
                               (data.flatten('F')).astype(float)))

    # execute
    try:
        skeleton.execute()

    except:
        # if the Java module fails, reraise the error it throws
        print("\n The underlying Java code did not execute cleanly: ")
        print(sys.exc_info()[0])
        raise
        return

    # Collect output
    medialImage_data = np.reshape(np.array(
                                    skeleton.getMedialSurfaceImage(),
                                    dtype=np.int8), dimensions, 'F')
    skelImage_data = np.reshape(np.array(
                                    skeleton.getMedialCurveImage(),
                                    dtype=np.int8), dimensions, 'F')


    # adapt header max for each image so that correct max is displayed
    # and create nifiti objects
 #   d_head['data_type'] = np.array(8).astype('int8') #convert the header as well
    header['cal_min'] = np.nanmin(medialImage_data)
    header['cal_max'] = np.nanmax(medialImage_data)
    medialImage = nb.Nifti1Image(medialImage_data, affine, header)

    header['cal_min'] = np.nanmin(skelImage_data)
    header['cal_max'] = np.nanmax(skelImage_data)
    skelImage = nb.Nifti1Image(skelImage_data, affine, header)

    if save_data:
        save_volume(os.path.join(output_dir, MedialSurface_file), medialImage)
        save_volume(os.path.join(output_dir, Medial_Curve_file), skelImage)

    return {'medialImage': medialImage, 'skelImage': skelImage}


