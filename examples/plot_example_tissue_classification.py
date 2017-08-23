"""
Tissue classification from MP2RAGE data
=======================================

This example shows how to obtain a tissue classification from MP2RAGE data
by performing the following steps:

1. Remove the skull and create a brain mask using
   :func:`nighres.brain.mp2rage_skullstripping`
2. Atlas-guided tissue classification using MGDM
   :func:`nighres.brain.mgdm_segmentation` [1]_
"""

############################################################################
# Import and download
# -------------------
# First we import ``nighres`` and the ``os`` module to set the output directory
# and names for the files we will download. Make sure to run this file in a
# directory you have write access to, or change the ``out_dir`` variable below.

import nighres
import os

out_dir = os.path.join(os.getcwd(), 'nighres_cache/tissue_classification')
t1map = os.path.join(out_dir, "T1map.nii.gz")
t1w = os.path.join(out_dir, "T1w.nii.gz")
inv2 = os.path.join(out_dir, "Inv2.nii.gz")

############################################################################
# We also try to import Nilearn plotting functions. If Nilearn is not
# installed, plotting will be skipped.
skip_plots = False
try:
    from nilearn import plotting
except ImportError:
    skip_plots = True

############################################################################
# Now we download an example MP2RAGE dataset
#
# .. todo:: download
inv2 = '/SCR/data/cbstools_testing/7t_trt/test_nii/INV2.nii.gz'
t1map = '/SCR/data/cbstools_testing/7t_trt/test_nii/T1map.nii.gz'
t1w = '/SCR/data/cbstools_testing/7t_trt/test_nii//T1w.nii.gz'

# nighres.download_from_url("http://openscience.cbs.mpg.de/bazin/7T_Quantitative/MP2RAGE-05mm/subject01_mp2rage_0p5iso_qT1.nii.gz",
#                           t1map)
# nighres.download_from_url("http://openscience.cbs.mpg.de/bazin/7T_Quantitative/MP2RAGE-05mm/subject01_mp2rage_0p5iso_uni.nii.gz",
#                           t1w)
############################################################################
# Skull stripping
# ----------------
# First we perform skull stripping. Only the second inversion image is required
# to calculate the brain mask. But if we input the T1map and T1w image as well,
# they will be masked for us. We also save the outputs in the ``out_dir``
# specified above and use a subject ID as the base file_name.
skullstripping_results = nighres.brain.mp2rage_skullstripping(
                                                        second_inversion=inv2,
                                                        t1_weighted=t1w,
                                                        t1_map=t1map,
                                                        save_data=True,
                                                        file_name='sub001',
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
    plotting.plot_roi(skullstripping_results['brain_mask'], t1w,
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')

############################################################################
# MGDM classification
# ---------------------
# Next, we use the masked data as input for tissue classification with the MGDM
# algorithm. MGDM works with a single contrast, but can  be improved with
# additional contrasts. In this case we use the T1-weigthed  image as well as
# the quantitative T1map.
mgdm_results = nighres.brain.mgdm_segmentation(
                        contrast_image1=skullstripping_results['t1w_masked'],
                        contrast_type1="Mp2rage7T",
                        contrast_image2=skullstripping_results['t1map_masked'],
                        contrast_type2="T1map7T",
                        save_data=True, file_name="sub001",
                        output_dir=out_dir)

############################################################################
# Now we look at some of the outputs from MGDM. The first is the
# topology-constrained segmentation

if not skip_plots:
    plotting.plot_img(mgdm_results['segmentation'], vmin=1, cmap='cubehelix',
                      annotate=False,  draw_cross=False)

############################################################################
# MGDM also creates an image which represents for each voxel the distance to
# its nearest border. It is useful to visualize partial voluming effects
if not skip_plots:
    plotting.plot_anat(mgdm_results['distance'], annotate=False,
                       draw_cross=False)

#############################################################################
# If the example is not run in a jupyter notebook, render the plots:
if not skip_plots:
    plotting.show()

#############################################################################
# References
# -----------
# .. [1] Bogovic, Prince and Bazin (2013). A multiple object geometric
#    deformable model for image segmentation. DOI: 10.1016/j.cviu.2012.10.006.A
#
# sphinx_gallery_thumbnail_number = 2
