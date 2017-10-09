# Plots are currently included as images, because example is too big to
# run on readthedocs servers
"""
Cortical depth estimation from a MGDM segmentation
=======================================

This example shows how to obtain a cortical laminar depth representation from a MGDM
segmentation result with the following steps:

1. Get a segmentation result from the 'tissue_classification' example
2. Extract the left, right, and cerebellar cortices with Extract Brain Region
   :func:`nighres.brain.extract_brain_region`
3. Cortical reconstruction with CRUISE
   :func:`nighres.cortex.cruise_cortex_extraction` [1]_
4. Anatomical depth estimation with Volumetric Layering
   :func:`nighres.laminar.volumetric_layering` [2]_
"""

############################################################################
# Import and point to previous example
# -------------------
# First we import ``nighres`` and the ``os`` module to set the output directory
# Make sure to run this file in a  directory you have write access to, or
# change the ``out_dir`` variable below.

import nighres
import os

in_dir = os.path.join(os.getcwd(), 'nighres_examples/tissue_classification')
out_dir = os.path.join(os.getcwd(), 'nighres_examples/cortical_depth_estimation')

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
# Now we pull the MGDM results from previous example
stripped = os.path.join(in_dir, 'sub001_sess1_strip_t1w.nii.gz')
segmentation = os.path.join(in_dir, 'sub001_sess1_mgdm_seg.nii.gz')
boundary_dist = os.path.join(in_dir, 'sub001_sess1_mgdm_dist.nii.gz')
max_labels = os.path.join(in_dir, 'sub001_sess1_mgdm_lbls.nii.gz')
max_probas = os.path.join(in_dir, 'sub001_sess1_mgdm_mems.nii.gz')

############################################################################
# Region Extraction
# ----------------
# Here we pull from the MGDM structures the needed regions for cortical
# reconstruction: the GM cortex ('region'), the underlying WM (with filled 
# subcortex and ventricles, 'inside'), the surrounding CSF (with masked regions,
# 'background')
left_cortex = nighres.brain.extract_brain_region(segmentation=segmentation,
                                            levelset_boundary=boundary_dist,
                                            maximum_membership=max_probas,
                                            maximum_label=max_labels,
                                            extracted_region='left_cerebrum'
                                            save_data=True,
                                            file_name='sub001_sess1',
                                            output_dir=out_dir)

right_cortex = nighres.brain.extract_brain_region(segmentation=segmentation,
                                            levelset_boundary=boundary_dist,
                                            maximum_membership=max_probas,
                                            maximum_label=max_labels,
                                            extracted_region='right_cerebrum'
                                            save_data=True,
                                            file_name='sub001_sess1',
                                            output_dir=out_dir)

cerebellum = nighres.brain.extract_brain_region(segmentation=segmentation,
                                            levelset_boundary=boundary_dist,
                                            maximum_membership=max_probas,
                                            maximum_label=max_labels,
                                            extracted_region='cerebellum'
                                            save_data=True,
                                            file_name='sub001_sess1',
                                            output_dir=out_dir)

############################################################################
# .. tip:: in Nighres functions that have several outputs return a
#    dictionary storing the different outputs. You can find the keys in the
#    docstring by typing ``nighres.brain.mp2rage_extract_brain_region?`` or list
#    them with ``left_cortex.keys()``
#
# To check if the extraction worked well we plot the GM and WM probabilities.
# You can also open the images stored in ``out_dir`` in
# your favourite interactive viewer and scroll through the volume.
#
# Like Nilearn, we use Nibabel SpatialImage objects to pass data internally.
# Therefore, we can directly plot the outputs using `Nilearn plotting functions
# <http://nilearn.github.io/plotting/index.html#different-plotting-functions>`_
# .

if not skip_plots:
    plotting.plot_roi(left_cortex['region_proba'], stripped,
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')
    plotting.plot_roi(left_cortex['inside_proba'], stripped,
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')
    plotting.plot_roi(right_cortex['region_proba'], stripped,
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')
    plotting.plot_roi(right_cortex['inside_proba'], stripped,
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')
    plotting.plot_roi(cerebellum['region_proba'], stripped,
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')
    plotting.plot_roi(cerebellum['inside_proba'], stripped,
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')
############################################################################
# .. image:: ../_static/cortical_depth_estimation1.png
# .. image:: ../_static/cortical_depth_estimation2.png
# .. image:: ../_static/cortical_depth_estimation3.png
#############################################################################

#############################################################################
# CRUISE cortical reconstruction
# ---------------------
# Next, we use the extracted data as input for cortex reconstruction with the CRUISE
# algorithm. CRUISE works with the membership functions as a guide and the WM inside
# mask as a (topologically spherical) starting point to grow a refined GM/WM boundary
# and CSF/GM boundary
cruise_left = nighres.cortex.cruise_cortex_extraction(
                        init_image=left_cortex['inside_mask'],
                        wm_image=left_cortex['inside_proba'],
                        gm_image=left_cortex['region_proba'],
                        csf_image=left_cortex['background_proba'],
                        normalize_probabilities=True,
                        save_data=True, file_name="sub001_sess1",
                        output_dir=out_dir)

cruise_right = nighres.cortex.cruise_cortex_extraction(
                        init_image=right_cortex['inside_mask'],
                        wm_image=right_cortex['inside_proba'],
                        gm_image=right_cortex['region_proba'],
                        csf_image=right_cortex['background_proba'],
                        normalize_probabilities=True,
                        save_data=True, file_name="sub001_sess1",
                        output_dir=out_dir)

cruise_cb = nighres.cortex.cruise_cortex_extraction(
                        init_image=cerebellum['inside_mask'],
                        wm_image=cerebellum['inside_proba'],
                        gm_image=cerebellum['region_proba'],
                        csf_image=cerebellum['background_proba'],
                        normalize_probabilities=True,
                        save_data=True, file_name="sub001_sess1",
                        output_dir=out_dir)

############################################################################
# Now we look at the topology-constrained segmentation CRUISE created
if not skip_plots:
    plotting.plot_img(cruise_left['cortex'],
                      vmin=1, vmax=50, cmap='cubehelix',  colorbar=True,
                      annotate=False,  draw_cross=False)
    plotting.plot_img(cruise_right['cortex'],
                      vmin=1, vmax=50, cmap='cubehelix',  colorbar=True,
                      annotate=False,  draw_cross=False)
    plotting.plot_img(cruise_cb['cortex'],
                      vmin=1, vmax=50, cmap='cubehelix',  colorbar=True,
                      annotate=False,  draw_cross=False)

############################################################################
# .. image:: ../_static/cortical_depth_estimation4.png
# .. image:: ../_static/cortical_depth_estimation5.png
# .. image:: ../_static/cortical_depth_estimation6.png
#############################################################################

#############################################################################
# Volumetric layering
# ---------------------
# Finally, we use the GM/WM boundary (GWB) and CSF/GM boundary (CGB) from CRUISE
# to compute cortical depth with a volume-preserving technique
depth_left = nighres.laminar.volumetric_layering(
                        inner_levelset=cruise_left['gwb'],
                        outer_levelset=cruise_left['cgb'],
                        n_layers=4,
                        save_data=True, file_name="sub001_sess1",
                        output_dir=out_dir)

depth_right = nighres.laminar.volumetric_layering(
                        inner_levelset=cruise_right['gwb'],
                        outer_levelset=cruise_right['cgb'],
                        n_layers=4,
                        save_data=True, file_name="sub001_sess1",
                        output_dir=out_dir)

depth_cb = nighres.laminar.volumetric_layering(
                        inner_levelset=cruise_cb['gwb'],
                        outer_levelset=cruise_cb['cgb'],
                        n_layers=4,
                        save_data=True, file_name="sub001_sess1",
                        output_dir=out_dir)

############################################################################
# Now we look at the lamianr depth estimates 
if not skip_plots:
    plotting.plot_img(depth_left['depth'],
                      vmin=1, vmax=50, cmap='autumn',  colorbar=True,
                      annotate=False,  draw_cross=False)
    plotting.plot_img(depth_right['depth'],
                      vmin=1, vmax=50, cmap='autumn',  colorbar=True,
                      annotate=False,  draw_cross=False)
    plotting.plot_img(depth_cb['depth'],
                      vmin=1, vmax=50, cmap='autumn',  colorbar=True,
                      annotate=False,  draw_cross=False)

############################################################################
# .. image:: ../_static/cortical_depth_estimation7.png
# .. image:: ../_static/cortical_depth_estimation8.png
# .. image:: ../_static/cortical_depth_estimation9.png
#############################################################################

#############################################################################
# If the example is not run in a jupyter notebook, render the plots:
if not skip_plots:
    plotting.show()

#############################################################################
# References
# -----------
# .. [1] Han et al (2004) CRUISE: Cortical Reconstruction Using Implicit 
#       Surface Evolution, NeuroImage, vol. 23, pp. 997--1012.
# .. [2] Waehnert et al (2014) Anatomically motivated modeling of cortical
#       laminae. DOI: 10.1016/j.neuroimage.2013.03.078
