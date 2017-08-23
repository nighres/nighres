"""
Sampling T1 at different intracortical depths
==============================================

This example shows how to sample T1 at different intracortical depth levels.
We start with a quantitative T1map and binary masks representing the pial and
white matter surface and perform the following steps:

1. Creating levelset representations of the pial and white matter surface
   :func:`nighres.surface.probability_to_levelset`
2. Equivolumetric layering of the cortical sheet
   :func:`nighres.laminar.volumetric_layering` [1]_
3. Sampling T1 on the different intracortical depth
   :func:`nighres.laminar.profile_sampling`
"""

############################################################################
# Import and download
# -------------------
# First we import ``nighres`` and the ``os`` module to set the output directory
# and filenames for the files we will download. Make sure to run this file in a
# directory you have write-access to, or change the ``out_dir`` variable below.

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
# Now we download ...
#
# .. todo:: download
t1map = '/SCR/data/cbstools_testing/7t_trt/test_nii/T1map.nii.gz'
t1w = '/SCR/data/cbstools_testing/7t_trt/test_nii//T1w.nii.gz'

# nighres.download_from_url("http://openscience.cbs.mpg.de/bazin/7T_Quantitative/MP2RAGE-05mm/subject01_mp2rage_0p5iso_qT1.nii.gz",
#                           t1map)
# nighres.download_from_url("http://openscience.cbs.mpg.de/bazin/7T_Quantitative/MP2RAGE-05mm/subject01_mp2rage_0p5iso_uni.nii.gz",
#                           t1w)
############################################################################
# To check if the skull stripping worked well, we plot the brain mask on top of
# the original image with slight transparency. You can also open the images
# stored in output_dir in an interactive viewer to scroll through the volume.
#
# Because we use NibabelSpatialImages to pass data internally, just like
# Nilearn, we can directly plot the outputs from Nighres functions using
# `Nilearn plotting functions
# <http://nilearn.github.io/plotting/index.html#different-plotting-functions
############################################################################
# Creating surfaces
# -----------------
# To create levelset representations of the pial and white matter surface,
# we first use the segmentation results to create binary masks representing
# those boundaries. This small thresholding loop bypasses a step that would
# usually be performed with CRUISE, an interface that has not yet been
# included in Nighres


############################################################################
# Now we use Nighres again to create the levelsets from the binary masks
gm_wm_levelset = nighres.surface.probability_to_levelset(
                                                    probability_image=wm_nii,
                                                    save_data=True,
                                                    file_name='gm_wm',
                                                    file_extension='nii.gz',
                                                    output_dir=out_dir)
gm_csf_levelset = nighres.surface.probability_to_levelset(
                                                    probability_image=gm_nii,
                                                    save_data=True,
                                                    file_name='gm_csf',
                                                    file_extension='nii.gz',
                                                    output_dir=out_dir)
############################################################################
# To check if the skull stripping worked well, we plot the brain mask on top of
# the original image with slight transparency. You can also open the images
# stored in output_dir in an interactive viewer to scroll through the volume.
#
# Because we use NibabelSpatialImages to pass data internally, just like
# Nilearn, we can directly plot the outputs from Nighres functions using
# `Nilearn plotting functions
# <http://nilearn.github.io/plotting/index.html#different-plotting-functions>`_
# .

###########################################################################
# Volumetric layering
# -------------------
# Once we have the levelset representations of the pial and white matter
# surface we can perform volume-preserving layering of the space between the
# two surfaces. Here we choose only 3 layers to save time.
layering_results = nighres.laminar.volumetric_layering(
                                                inner_levelset=gm_wm_levelset,
                                                outer_levelset=gm_csf_levelset,
                                                n_layers=3,
                                                save_data=True,
                                                output_dir=out_dir)

###########################################################################
# Sampling T1
# -------------
# Finally, we use the intracortical layers, represented as levelsets,
# to sample T1 across the different cortical depth levels
profiles = nighres.laminar.profile_sampling(
                        profile_surface_image=layering_results['boundaries'],
                        intensity_image=t1map,
                        save_data=True,
                        output_dir=out_dir)

#############################################################################
#
# .. todo:: Visualize data using Nilearn

#############################################################################
# References
# -----------
# .. [1] Waehnert et al (2014). Anatomically motivated modeling of cortical
#    laminae. DOI: 10.1016/j.neuroimage.2013.03.078
