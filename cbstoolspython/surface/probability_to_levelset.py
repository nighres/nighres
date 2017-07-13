import os
import numpy as np
import nibabel as nb
import cbstools
from ..io import load_volume, save_volume


def probability_to_levelset(tissue_prob_img, save_data=True, base_name=None):

    '''
    Creates levelset surface representations from a tissue classification.

        Parameters
        -----------
        tissue_prob_img : Tissue segmentation to be turned into levelset.
            Either a binary tissue classfication with value 1 inside and 0
            outside the to-be-created surface, or ????
            Can be a path to a Nifti file or Nibabel image object.
        save_data : Whether the output levelset image should be saved
            (default is 'True').
        base_name : If save_data is set to True, this parameter can be used to
            specify where the output should be saved. Thus can be the path to a
            directory or a full filename. The suffix 'levelset' will be added
            to the filename. If None (default), the output will be saved to the
            current directory.

        Returns
        -------
        Levelset representation of surface as Nibabel image object
    '''

    # load the data as well as filenames and headers for saving later
    prob_img = load_volume(tissue_prob_img)
    prob_data = prob_img.get_data()
    hdr = prob_img.get_header()
    aff = prob_img.get_affine()

    try:
        cbstools.initVM(initialheap='6000m', maxheap='6000m')
    except ValueError:
        pass

    prob2level = cbstools.SurfaceProbabilityToLevelset()
    prob2level.setProbabilityImage(cbstools.JArray('float')((prob_data.flatten('F')).astype(float)))
    prob2level.setDimensions(prob_data.shape)
    zooms = [x.item() for x in hdr.get_zooms()]
    prob2level.setResolutions(zooms[0], zooms[1], zooms[2])
    prob2level.execute()

    levelset_data = np.reshape(np.array(prob2level.getLevelSetImage(),
                               dtype=np.float32), prob_data.shape, 'F')

    levelset_img = nb.Nifti1Image(levelset_data, aff, hdr)

    if save_data:
        if base_name:
            base_name += '_'
        else:
            if not isinstance(tissue_prob_img, basestring):
                base_name = os.getcwd() + '/'
                print "saving to %s" % base_name
            else:
                dir_name = os.path.dirname(tissue_prob_img)
                base_name = os.path.basename(tissue_prob_img)
                base_name = os.path.join(dir_name,
                                         base_name[:base_name.find('.')]) + '_'
                print "saving to %s" % base_name

        save_volume(base_name+'levelset.nii.gz', levelset_img)

    return levelset_img
