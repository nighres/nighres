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
    if subject_id is 'sub001_sess1':
        file_sources = [nitrc + x for x in ['10234', '10235', '10236']]
    elif subject_id is 'sub002_sess1':
        file_sources = [nitrc + x for x in ['10852', '10853', '10854']]
    elif subject_id is 'sub003_sess1':
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
           
           
def download_DOTS_atlas(data_dir, overwrite=False):
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
        * fiber_dir : path to atlas direction mask

    References
    ----------
    .. [1] Bazin et al (2011). Direct segmentation of the major white matter 
           tracts in diffusion tensor images.
           DOI: 10.1016/j.neuroimage.2011.06.020
    """

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
           
