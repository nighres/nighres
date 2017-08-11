import nibabel as nb
import numpy as np


def load_volume(volume):
    """
    Load volumetric data into a
    `Nibabel SpatialImage <http://nipy.org/nibabel/reference/nibabel.spatialimages.html#nibabel.spatialimages.SpatialImage>`_

    Parameters
    ----------
    volume: niimg
        Volumetric data to be loaded, can be a path to a file that nibabel can
        load, or a Nibabel SpatialImage

    Returns
    ----------
    image: Nibabel SpatialImage

    Notes
    ----------
    Originally created as part of Laminar Python [1]_ .

    References
    -----------
    .. [1] Huntenburg et al. (2017), Laminar Python: Tools for cortical
       depth-resolved analysis of high-resolution brain imaging data in
       Python. DOI: 10.3897/rio.3.e12346
    """  # noqa

    # if input is a filename, try to load it
    if isinstance(volume, basestring):
        # importing nifti files
        image = nb.load(volume)
    # if volume is already a nibabel object
    elif isinstance(volume, nb.spatialimages.SpatialImage):
        image = volume
    else:
        raise ValueError('Input volume must be a either a path to a file in a '
                         'that Nibabel can load, or a nibabel SpatialImage.')
    return image


def save_volume(filename, volume, dtype='float32', overwrite_file=True):
    """
    Save volumetric data that is a
    `Nibabel SpatialImage <http://nipy.org/nibabel/reference/nibabel.spatialimages.html#nibabel.spatialimages.SpatialImage>`_
    to a file

    Parameters
    ----------
    filename: str
        Full path and filename under which volume should be saved. The
        extension determines the file format (must be supported by Nibabel)
    volume: Nibabel SpatialImage
        Volumetric data to be saved
    dtype: str, optional
        Datatype in which volumetric data should be stored (default is float32)
    overwrite_file: bool, optional
        Overwrite existing files (default is True)

    Notes
    ----------
    Originally created as part of Laminar Python [1]_ .

    References
    -----------
    .. [1] Huntenburg et al. (2017), Laminar Python: Tools for cortical
       depth-resolved analysis of high-resolution brain imaging data in
       Python. DOI: 10.3897/rio.3.e12346
    """  # noqa
    import os
    if dtype is not None:
        volume.set_data_dtype(dtype)
    if os.path.isfile(filename) and overwrite_file is False:
        print("\nThis file exists and overwrite_file was set to False, "
              "file not saved.")
    else:
        try:
            volume.to_filename(filename)
            print("\nSaving {0}").format(filename)
        except AttributeError:
            print('\nInput volume must be a Nibabel SpatialImage.')
