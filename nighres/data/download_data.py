import os
from urllib.request import urlretrieve
from nighres.global_settings import ATLAS_DIR

def download_7T_TRT(data_dir, overwrite=False, subject_id='sub001_sess1'):
    """
    Downloads the MP2RAGE data from the 7T Test-Retest
    dataset published by Gorgolewski et al (2015) [1]_

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded files should be stored. A
        subdirectory called '7T_TRT' will be created in this location.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)
    subject_id: 'sub001_sess1', 'sub002_sess1', 'sub003_sess1'}
        Which dataset to download (default is 'sub001_sess1')

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * inv2 : path to second inversion image
        * t1w : path to T1-weighted (uniform) image
        * t1map : path to quantitative T1 image

    Notes
    ----------
    The full dataset is available at http://openscience.cbs.mpg.de/7t_trt/

    References
    ----------
    .. [1] Gorgolewski et al (2015). A high resolution 7-Tesla resting-state
       fMRI test-retest dataset with cognitive and physiological measures.
       DOI: 10.1038/sdata.2014.
    """

    data_dir = os.path.join(data_dir, '7T_TRT')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    nitrc = 'https://www.nitrc.org/frs/download.php/'
    if subject_id == 'sub001_sess1':
        file_sources = [nitrc + x for x in ['10234', '10235', '10236']]
    elif subject_id == 'sub002_sess1':
        file_sources = [nitrc + x for x in ['10852', '10853', '10854']]
    elif subject_id == 'sub003_sess1':
        file_sources = [nitrc + x for x in ['10855', '10856', '10857']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    [subject_id+'_INV2.nii.gz',
                     subject_id+'_T1map.nii.gz',
                     subject_id+'_T1w.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'inv2': file_targets[0],
            't1map': file_targets[1],
            't1w': file_targets[2]}


def download_DTI_2mm(data_dir, overwrite=False):
    """
    Downloads an example DTI data set

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded files should be stored. A
        subdirectory called 'DTI_2mm' will be created in this location.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * dti : path to DTI image
        * mask : path to binary brain mask

    """

    data_dir = os.path.join(data_dir, 'DTI_2mm')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    nitrc = 'https://www.nitrc.org/frs/download.php/'

    file_sources = [nitrc + x for x in ['11511', '11512']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['DTI_2mm.nii.gz',
                     'DTI_2mm_brain_mask.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'dti': file_targets[0],
            'mask': file_targets[1]}


def download_DOTS_atlas(data_dir=None, overwrite=False):
    """
    Downloads the statistical atlas presented in [1]_

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded atlas files should be stored. A
        subdirectory called 'DOTS_atlas' will be created in this location.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * fiber_p : path to atlas probability image
        * fiber_dir : path to atlas direction image

    References
    ----------
    .. [1] Bazin et al (2011). Direct segmentation of the major white matter
           tracts in diffusion tensor images.
           DOI: 10.1016/j.neuroimage.2011.06.020
    """

    if (data_dir is None):
        data_dir = ATLAS_DIR

    data_dir = os.path.join(data_dir, 'DOTS_atlas')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    nitrc = 'https://www.nitrc.org/frs/download.php/'

    file_sources = [nitrc + x for x in ['11514', '11513']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['fiber_p.nii.gz',
                     'fiber_dir.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'fiber_p': file_targets[0],
            'fiber_dir': file_targets[1]}

def download_MASSP_atlas(data_dir=None, overwrite=False):
    """
    Downloads the MASSP atlas presented in [1]_

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded atlas files should be stored. A
        subdirectory called 'massp-prior' will be created in this location.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * histograms : path to histogram image
        * spatial_probas : path to spatial probability image
        * spatial_labels : path to spatial label image
        * skeleton_probas : path to skeleton probability image
        * skeleton_labels : path to skeleton label image

    References
    ----------
    .. [1] Bazin et al (2020). Multi-contrast Anatomical Subcortical
    Structure Parcellation. Under review.
    """

    if (data_dir is None):
        data_dir = ATLAS_DIR

    data_dir = os.path.join(data_dir, 'massp-prior')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    figshare = 'https://uvaauas.figshare.com/ndownloader/files/'

    file_sources = [figshare + x for x in
                    ['22627481','22627484','22627475','22627478','22627472']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['massp_17structures_spatial_label.nii.gz',
                     'massp_17structures_spatial_proba.nii.gz',
                     'massp_17structures_skeleton_label.nii.gz',
                     'massp_17structures_skeleton_proba.nii.gz',
                     'massp_17structures_r1r2sqsm_histograms.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'spatial_labels': file_targets[0],
            'spatial_probas': file_targets[1],
            'skeleton_labels': file_targets[2],
            'skeleton_probas': file_targets[3],
            'histograms': file_targets[4]}

def download_MP2RAGEME_sample(data_dir, overwrite=False):
    """
    Downloads an example data set from a MP2RAGEME acquisition _[1].

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded atlas files should be stored.
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * qr1 : path to quantitative R1 map image
        * qr2s : path to quantitative R2* map image
        * qsm : path to QSM image

    References
    ----------
    .. [1] Caan et al (2018). MP2RAGEME: T1, T2*, and QSM mapping in one
    sequence at 7 tesla. doi:10.1002/hbm.24490
    """

    data_dir = os.path.join(data_dir, 'mp2rageme')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    figshare = 'https://uvaauas.figshare.com/ndownloader/files/'

    file_sources = [figshare + x for x in
                    ['22678334','22678337','22628750']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['sample-subject_mp2rageme-qr1_brain.nii.gz',
                     'sample-subject_mp2rageme-qr2s_brain.nii.gz',
                     'sample-subject_mp2rageme-qsm_brain.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'qr1': file_targets[0],
            'qr2s': file_targets[1],
            'qsm': file_targets[2]}

def download_AHEAD_template(data_dir=None, overwrite=False):
    """
    Downloads the AHEAD group template _[1].

    Parameters
    ----------
    data_dir: str
        Writeable directory in which downloaded atlas files should be stored. A
        subdirectory called 'ahead-template' will be created in this location
        (default is ATLAS_DIR)
    overwrite: bool
        Overwrite existing files in the same exact path (default is False)

    Returns
    ----------
    dict
        Dictionary with keys pointing to the location of the downloaded files

        * qr1 : path to quantitative R1 map image
        * qr2s : path to quantitative R2* map image
        * qsm : path to QSM image

    References
    ----------
    .. [1] Alkemade et al (under review). The Amsterdam Ultra-high field adult
       lifespan database (AHEAD): A freely available multimodal 7 Tesla
       submillimeter magnetic resonance imaging database.
    """

    if (data_dir is None):
        data_dir = ATLAS_DIR

    data_dir = os.path.join(data_dir, 'ahead-template')

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    figshare = 'https://uvaauas.figshare.com/ndownloader/files/'

    file_sources = [figshare + x for x in
                    ['22679537','22679543','22679546']]

    file_targets = [os.path.join(data_dir, filename) for filename in
                    ['ahead_med_qr1.nii.gz',
                     'ahead_med_qr2s.nii.gz',
                     'ahead_med_qsm.nii.gz']]

    for source, target in zip(file_sources, file_targets):

        if os.path.isfile(target) and overwrite is False:
            print("\nThe file {0} exists and overwrite was set to False "
                  "-- not downloading.".format(target))
        else:
            print("\nDownloading to {0}".format(target))
            urlretrieve(source, target)

    return {'qr1': file_targets[0],
            'qr2s': file_targets[1],
            'qsm': file_targets[2]}
