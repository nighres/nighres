# Plots are currently included as images, because example is too big to
# run on readthedocs servers
"""
Vascular Reconstruction
========================

This example shows how to perform vascular reconstruction based on MP2RAGE
data by performing the following steps:

1. Downloading three open MP2RAGE datasets using
    :func:`nighres.data.download_7T_TRT` [1]_
2. Remove the skull and create a brain mask using
    :func:`nighres.brain.mp2rage_skullstripping`
3. Vasculature reconstruction using
   :func:`nighres.filtering.multiscale_vessel_filter` [2]_

Important note: this example extracts arteries as bright vessels in a T1-weighted
7T scan, instead of the veins extracted from QSM in [2]_, because these data sets
could not be made openly available. Processing of QSM images would however follow
the same pipeline
"""

############################################################################
# Import and download
# -------------------
# First we import ``nighres`` and the ``os`` module to set the output directory
# Make sure to run this file in a  directory you have write access to, or
# change the ``out_dir`` variable below.

import nighres
import os
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt

in_dir = os.path.join(os.getcwd(), 'nighres_examples/data_sets')
out_dir = os.path.join(os.getcwd(), 'nighres_examples/vascular_reconstruction')

############################################################################
# We also try to import Nilearn plotting functions. If Nilearn is not
# installed, plotting will be skipped.
skip_plots = False
try:
    from nilearn import plotting
except ImportError:
    skip_plots = True
    print('Nilearn could not be imported, plotting will be skipped')

############################################################################
# Now we download an example MP2RAGE dataset. It is the structural scan of the
# first subject, first session of the 7T Test-Retest dataset published by
# Gorgolewski et al (2015) [1]_.
dataset = nighres.data.download_7T_TRT(in_dir, subject_id='sub001_sess1')
############################################################################
# Skull stripping
# ----------------
# First we perform skull stripping. Only the second inversion image is required
# to calculate the brain mask. But if we input the T1map and T1w image as well,
# they will be masked for us. We also save the outputs in the ``out_dir``
# specified above and use a subject ID as the base file_name.
skullstripping_results = nighres.brain.mp2rage_skullstripping(
                                            second_inversion=dataset['inv2'],
                                            t1_weighted=dataset['t1w'],
                                            t1_map=dataset['t1map'],
                                            save_data=True,
                                            file_name='sub001_sess1',
                                            output_dir=out_dir)


############################################################################
# .. tip:: in Nighres functions that have several outputs return a
#    dictionary storing the different outputs. You can find the keys in the
#    docstring by typing ``nighres.brain.mp2rage_skullstripping?`` or list
#    them with ``skullstripping_results.keys()``
#
# To check if the skull stripping worked well we plot the brain mask on top of
# the original image. You can also open the images stored in ``out_dir`` in
# your favourite interactive viewer and scroll through the volume.
#
# Like Nilearn, we use Nibabel SpatialImage objects to pass data internally.
# Therefore, we can directly plot the outputs using `Nilearn plotting functions
# <http://nilearn.github.io/plotting/index.html#different-plotting-functions>`_
# .

if not skip_plots:
    plotting.plot_roi(skullstripping_results['brain_mask'], dataset['t1map'],
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')
############################################################################


#############################################################################
# Vessel reconstruction
# ---------------------
# Next, we use the vessel filter to estimate the vasculature from the QSM data
vessel_result = nighres.filtering.multiscale_vessel_filter(
                        input_image=skullstripping_results['t1w_masked'],
                        scales=2,
                        save_data=True, file_name="sub001_sess1",
                        output_dir=out_dir)


############################################################################
# Now we look at the topology-constrained segmentation MGDM created
if not skip_plots:
    plotting.plot_img(vessel_result['pv'],
                      vmin=0, vmax=1, cmap='cubehelix',  colorbar=True,
                      annotate=False,  draw_cross=False)
    plotting.plot_img(vessel_result['diameter'],
                      vmin=0, vmax=4, cmap='cubehelix',  colorbar=True,
                      annotate=False,  draw_cross=False)
############################################################################


#############################################################################
# If the example is not run in a jupyter notebook, render the plots:
if not skip_plots:
    plotting.show()

############################################################################
# Additional visualization: compute maximum intensity projections
data = nighres.io.load_volume(vessel_result['pv']).get_fdata()
fig, ax = plt.subplots(1, 3, figsize=(28,5))
ax[0].imshow(np.rot90(np.max(data[100:130,:,:], axis=0)), cmap = 'gray')
ax[1].imshow(np.rot90(np.max(data[:,170:200,:], axis=1)), cmap = 'gray')
ax[2].imshow(np.rot90(np.max(data[:,:,170:200], axis=2)), cmap = 'gray')
for i in range(3):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
fig.tight_layout()
#fig.savefig('segmentation.png')
plt.show()


#############################################################################
# References
# -----------
# .. [1] Gorgolewski et al (2015). A high resolution 7-Tesla resting-state fMRI
#    test-retest dataset with cognitive and physiological measures.
#    DOI: 10.1038/sdata.2014.54
# .. [2] Huck et al. (2019) High resolution atlas of the venous brain vasculature 
#    from 7 T quantitative susceptibility maps. DOI: 10.1007/s00429-019-01919-4
