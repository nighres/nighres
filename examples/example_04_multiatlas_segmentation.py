# Plots are currently included as images, because example is too big to
# run on readthedocs servers
"""
Multiatlas Segmentation
========================

This example shows how to perform multi-atlas segmentation based on MP2RAGE
data by performing the following steps:

1. Downloading three open MP2RAGE datasets using
    :func:`nighres.data.download_7T_TRT`
2. Remove the skull and create a brain mask using
    :func:`nighres.brain.mp2rage_skullstripping`
3. Atlas-guided tissue classification using MGDMfor first two subjects to
    be used as an atlas using :func:`nighres.brain.mgdm_segmentation` [1]_
4. Co-register non-linearly the atlas brains the the third subject using
    :func:`nighres.registration.embedded_antspy` [2]_
5. Deform segmentation labels using
    :func:`nighres.registration.apply_deformation`
6. Turn individual labels into levelset surfaces using
    :func:`nighres.surface.probability_to_levelset`
7. Build a final shape average using
    :func: `nighres.shape.levelset_fusion`

Important note: this example is both computationally expensive (recomputing
everything from basic inputs) and practically pointless (a direct MGDM
segmentation or a multi-atlas approach with manually defined labels and more
subjects would both be meaningful). This example is only meant as illustration.
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

in_dir = os.path.join(os.getcwd(), 'nighres_examples/data_sets')
out_dir = os.path.join(os.getcwd(), 'nighres_examples/multiatlas_segmentation')

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
# Gorgolewski et al (2015) [3]_.
dataset1 = nighres.data.download_7T_TRT(in_dir, subject_id='sub001_sess1')
dataset2 = nighres.data.download_7T_TRT(in_dir, subject_id='sub002_sess1')
dataset3 = nighres.data.download_7T_TRT(in_dir, subject_id='sub003_sess1')
############################################################################
# Skull stripping
# ----------------
# First we perform skull stripping. Only the second inversion image is required
# to calculate the brain mask. But if we input the T1map and T1w image as well,
# they will be masked for us. We also save the outputs in the ``out_dir``
# specified above and use a subject ID as the base file_name.
skullstripping_results1 = nighres.brain.mp2rage_skullstripping(
                                            second_inversion=dataset1['inv2'],
                                            t1_weighted=dataset1['t1w'],
                                            t1_map=dataset1['t1map'],
                                            save_data=True,
                                            file_name='sub001_sess1',
                                            output_dir=out_dir)

skullstripping_results2 = nighres.brain.mp2rage_skullstripping(
                                            second_inversion=dataset2['inv2'],
                                            t1_weighted=dataset2['t1w'],
                                            t1_map=dataset2['t1map'],
                                            save_data=True,
                                            file_name='sub002_sess1',
                                            output_dir=out_dir)

skullstripping_results3 = nighres.brain.mp2rage_skullstripping(
                                            second_inversion=dataset3['inv2'],
                                            t1_weighted=dataset3['t1w'],
                                            t1_map=dataset3['t1map'],
                                            save_data=True,
                                            file_name='sub003_sess1',
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
    plotting.plot_roi(skullstripping_results1['brain_mask'], dataset1['t1map'],
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')
    plotting.plot_roi(skullstripping_results2['brain_mask'], dataset2['t1w'],
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')
    plotting.plot_roi(skullstripping_results3['brain_mask'], dataset3['t1w'],
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')
############################################################################


#############################################################################
# MGDM classification
# ---------------------
# Next, we use MGDM to estimate anatomical labels from subjects 1 and 2
mgdm_results1 = nighres.brain.mgdm_segmentation(
                        contrast_image1=skullstripping_results1['t1w_masked'],
                        contrast_type1="Mp2rage7T",
                        contrast_image2=skullstripping_results1['t1map_masked'],
                        contrast_type2="T1map7T",
                        save_data=True, file_name="sub001_sess1",
                        output_dir=out_dir)

mgdm_results2 = nighres.brain.mgdm_segmentation(
                        contrast_image1=skullstripping_results2['t1w_masked'],
                        contrast_type1="Mp2rage7T",
                        contrast_image2=skullstripping_results2['t1map_masked'],
                        contrast_type2="T1map7T",
                        save_data=True, file_name="sub002_sess1",
                        output_dir=out_dir)

############################################################################
# Now we look at the topology-constrained segmentation MGDM created
if not skip_plots:
    plotting.plot_img(mgdm_results1['segmentation'],
                      vmin=1, vmax=50, cmap='cubehelix',  colorbar=True,
                      annotate=False,  draw_cross=False)
    plotting.plot_img(mgdm_results2['segmentation'],
                      vmin=1, vmax=50, cmap='cubehelix',  colorbar=True,
                      annotate=False,  draw_cross=False)
    plotting.show()
############################################################################

#############################################################################
# SyN co-registration
# ---------------------
# Next, we use the masked data as input for co-registration. The T1 maps are
# used here as they are supposed to be more similar
ants_results1 = nighres.registration.embedded_antspy(
                        source_image=skullstripping_results1['t1map_masked'],
                        target_image=skullstripping_results3['t1map_masked'],
                        run_rigid=True, run_affine=False, run_syn=False,
                        coarse_iterations=40,
                        medium_iterations=0, fine_iterations=0,
                        cost_function='MutualInformation',
                        interpolation='NearestNeighbor',
                        ignore_affine=True,
                        save_data=True, file_name="sub001_sess1",
                        output_dir=out_dir)

ants_results2 = nighres.registration.embedded_antspy(
                        source_image=skullstripping_results2['t1map_masked'],
                        target_image=skullstripping_results3['t1map_masked'],
                        run_rigid=True, run_affine=False, run_syn=False,
                        coarse_iterations=40,
                        medium_iterations=0, fine_iterations=0,
                        cost_function='MutualInformation',
                        interpolation='NearestNeighbor',
                        ignore_affine=True,
                        save_data=True, file_name="sub002_sess1",
                        output_dir=out_dir)

############################################################################
# Now we look at the coregistered image that SyN created
if not skip_plots:
    plotting.plot_img(ants_results1['transformed_source'],
                      annotate=False,  draw_cross=False)
    plotting.plot_img(ants_results2['transformed_source'],
                      annotate=False,  draw_cross=False)
############################################################################

#############################################################################
# Apply deformations to segmentations
# ------------------------------------
# We use the computed deformation to transform MGDM segmentations
deformed1 = nighres.registration.apply_coordinate_mappings(
                        image=mgdm_results1['segmentation'],
                        mapping1=ants_results1['mapping'],
                        save_data=True, file_name="sub001_sess1_seg",
                        output_dir=out_dir)

deformed2 = nighres.registration.apply_coordinate_mappings(
                        image=mgdm_results2['segmentation'],
                        mapping1=ants_results2['mapping'],
                        save_data=True, file_name="sub002_sess1_seg",
                        output_dir=out_dir)

############################################################################
# Now we look at the segmentations deformed by SyN
if not skip_plots:
    plotting.plot_img(deformed1['result'],
                      annotate=False,  draw_cross=False)
    plotting.plot_img(deformed2['result'],
                      annotate=False,  draw_cross=False)

    plotting.show()
############################################################################

#############################################################################
# Transform a selected label into levelset representation
# ---------------------------------------------------------
# We use the deformed MGDM segmentations

# label 32 = left caudate
img1 = nighres.io.load_volume(deformed1['result'])
struct1 = nb.Nifti1Image((img1.get_fdata()==32).astype(float),
                            img1.affine, img1.header)

img2 = nighres.io.load_volume(deformed2['result'])
struct2 = nb.Nifti1Image((img2.get_fdata()==32).astype(float),
                            img2.affine, img2.header)

levelset1 = nighres.surface.probability_to_levelset(
                        probability_image=struct1,
                        save_data=True, file_name="sub001_sess1_struct",
                        output_dir=out_dir)

levelset2 = nighres.surface.probability_to_levelset(
                        probability_image=struct2,
                        save_data=True, file_name="sub002_sess1_struct",
                        output_dir=out_dir)

final_seg = nighres.shape.levelset_fusion(levelset_images=[levelset1['result'],
                        levelset2['result']],
                        correct_topology=True,
                        save_data=True, file_name="sub003_sess1_struct_seg",
                        output_dir=out_dir)

############################################################################
# Now we look at the final segmentation from shape fusion
if not skip_plots:
    img = nighres.io.load_volume(levelset1['result'])
    mask = nb.Nifti1Image((img.get_fdata()<0).astype(int),
                                img.affine, img.header)
    plotting.plot_roi(mask, dataset3['t1map'],
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')

    img = nighres.io.load_volume(levelset2['result'])
    mask = nb.Nifti1Image((img.get_fdata()<0).astype(int),
                                img.affine, img.header)
    plotting.plot_roi(mask, dataset3['t1map'],
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')

    img = nighres.io.load_volume(final_seg['result'])
    mask = nb.Nifti1Image((img.get_fdata()<0).astype(int),
                                img.affine, img.header)
    plotting.plot_roi(mask, dataset3['t1map'],
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='autumn')

############################################################################



#############################################################################
# If the example is not run in a jupyter notebook, render the plots:
if not skip_plots:
    plotting.show()

#############################################################################
# References
# -----------
# .. [1] Bogovic, Prince and Bazin (2013). A multiple object geometric
#    deformable model for image segmentation. DOI: 10.1016/j.cviu.2012.10.006.A
# .. [2] Avants et al (2008). Symmetric diffeomorphic image registration with
#    cross-correlation: evaluating automated labeling of elderly and
#    neurodegenerative brain. DOI: 10.1016/j.media.2007.06.004
# .. [3] Gorgolewski et al (2015). A high resolution 7-Tesla resting-state fMRI
#    test-retest dataset with cognitive and physiological measures.
#    DOI: 10.1038/sdata.2014.54
