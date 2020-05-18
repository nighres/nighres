import numpy as np
import nibabel as nb
import os
import sys
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving


def filter_stacking(dura_img=None, pvcsf_img=None, arteries_img=None,
                           save_data=False, overwrite=False, output_dir=None,
                           file_name=None):
    """ Filter stacking

    A small utility to combine multiple priors derived from filtering of
    the CSF partial voluming, arteries, and/or remaining dura mater. The filter
    priors (in [0,1]) are arranged in a specific way (in increments of 2) that
    is expected by the Filters contrast type in MGDM

    Parameters
    ----------
    dura_img: niimg, optional
        Prior for the location of remaining dura mater after skull stripping.
        At least one prior image is required
    pvcsf_img: niimg, optional
        Prior for the location of CSF partial voluming (mostly in sulcal
        regions). At least one prior image is required
    arteries_img: niimg, optional
        Prior for the location of arteries, visible e.g. in MP2RAGE images.
        At least one prior image is required
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
        (suffix in brackets)

        * result (niimg): Combined image, where only the strongest priors
          are kept (_bfs-img)


    Notes
    ----------
    Original Java module by Pierre-Louis Bazin.
    """

    print('\nFilter stacking')

    # check if there's inputs
    if (dura_img is None and pvcsf_img is None and arteries_img is None):
        raise ValueError('You must specify at least one of '
                         'pvcsf_img, arteries_img and dura_img')

    # find the first existing input for dimensions, resolutions, name
    img = None
    if (dura_img != None): img = dura_img
    elif (pvcsf_img != None): img = pvcsf_img
    elif (arteries_img != None): img = arteries_img

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, img)

        filter_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                   rootfile=img,
                                   suffix='bfs-img'))
        if overwrite is False \
            and os.path.isfile(filter_file) :

            print("skip computation (use existing results)")
            output = {"result": filter_file}
            return output

    affine = load_volume(img).affine
    header = load_volume(img).header
    resolution = [x.item() for x in header.get_zooms()]
    dimensions = header.get_data_shape()
    nx = dimensions[0];
    ny = dimensions[1];
    nz = dimensions[2];

    # build empty filter
    filter_data = np.zeros(dimensions)

    # add inputs
    if dura_img is not None:
        dura_data = load_volume(dura_img).get_data()
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    filter_data[x,y,z] = dura_data[x,y,z]

        dura_data = None
        dura_img = None

    if pvcsf_img is not None:
        pvcsf_data = load_volume(pvcsf_img).get_data()
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if (pvcsf_data[x,y,z]>filter_data[x,y,z]):
                        filter_data[x,y,z] = pvcsf_data[x,y,z]+2.0

        pvcsf_data = None
        pvcsf_img = None

    if arteries_img is not None:
        arteries_data = load_volume(arteries_img).get_data()
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if (filter_data[x,y,z]>2):
                        if (arteries_data[x,y,z]>filter_data[x,y,z]-2):
                            filter_data[x,y,z] = arteries_data[x,y,z]+4.0
                    else:
                         if (arteries_data[x,y,z]>filter_data[x,y,z]):
                            filter_data[x,y,z] = arteries_data[x,y,z]+4.0

        arteries_data = None
        arteries_img = None

    header['cal_max'] = np.nanmax(filter_data)
    filter_img = nb.Nifti1Image(filter_data, affine, header)

    if save_data:
        save_volume(filter_file, filter_img)
        return {"result": filter_file}
    else:
        return {"result": filter_img}
