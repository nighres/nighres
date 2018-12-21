import os
from urllib.request import urlretrieve


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
