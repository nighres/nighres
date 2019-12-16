import os
import numpy as np
import nibabel as nb
import nighresjava
from ..io import load_volume, save_volume
from ..utils import _output_dir_4saving, _fname_4saving


def levelset_to_probability(levelset_image, distance_mm=5,
                            save_data=False, overwrite=False, output_dir=None,
                            file_name=None):

    """Levelset to probability

    Creates a probability map from the distance to a levelset surface
    representation.

    Parameters
    ----------
    levelset_image: niimg
        Levelset image to be turned into probabilities
    distance_mm: float, optional
        Distance used for the range of probabilities in ]0,1[
    save_data: bool, optional
        Save output data to file (default is False)
    overwrite: bool, optional
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

        * result (niimg): Probability map (output file suffix _l2p-proba)

    Notes
    ----------
    Ported from original Java module by Pierre-Louis Bazin
    """

    print("\nLevelset to Probabilities")

    # make sure that saving related parameters are correct
    if save_data:
        output_dir = _output_dir_4saving(output_dir, levelset_image)

        proba_file = os.path.join(output_dir,
                        _fname_4saving(module=__name__,file_name=file_name,
                                       rootfile=levelset_image,
                                       suffix='l2p-proba'))

        if overwrite is False \
            and os.path.isfile(proba_file) :

            print("skip computation (use existing results)")
            output = {'result': proba_file}
            return output

    # load the data
    lvl_img = load_volume(levelset_image)
    lvl_data = lvl_img.get_data()
    hdr = lvl_img.header
    aff = lvl_img.affine
    resolution = [x.item() for x in hdr.get_zooms()]
    dimensions = lvl_data.shape

    # algorithm
    distance = distance_mm/np.min(resolution)
    print("voxel spread: "+str(distance))
    p_data = 0.5*(np.maximum(-1,np.minimum(1,-lvl_data/distance))+1)

    hdr['cal_min'] = np.nanmin(p_data)
    hdr['cal_max'] = np.nanmax(p_data)
    proba = nb.Nifti1Image(p_data, aff, hdr)

    if save_data:
        save_volume(proba_file, proba)
        return {'result': proba_file}
    else:
        return {'result': proba}
