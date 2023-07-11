# Plots are currently included as images, because example is too big to
# run on readthedocs servers
"""
Brain co-registration from MP2RAGE data
========================================

This example shows how to co-register MP2RAGE data
by performing the following steps:

1. Download two downsampled MP2RAGEME dataset using 
    :func:`nighres.data.download_MP2RAGEME_testdata` [1]_
2. Remove the skull and create a brain mask using
    :func:`nighres.brain.intensity_based_skullstripping` [2]_
3. Co-register non-linearly the brains using
    :func:`nighres.registration.embedded_antspy` [3]_
4. Deform additional contrasts using
    :func:`nighres.registration.apply_deformation`
"""

############################################################################
# Import and download
# -------------------
# First we import ``nighres`` and the ``os`` module to set the output directory
# Make sure to run this file in a  directory you have write access to, or
# change the ``out_dir`` variable below.

import nighres
import os

in_dir = os.path.join(os.getcwd(), 'nighres_testing/data_sets')
out_dir = os.path.join(os.getcwd(), 'nighres_testing/brain_registration')
os.makedirs(out_dir, exist_ok=True)

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
# Now we download an example MP2RAGEME dataset, which consists of a MPRAGE sequence
# with two inversions interleaved and multiple echoes on the second inversion [1]_.
# The data includes both a full brain scan and an imaging slab with higher resolution,
# which we will coregister.
dataset = nighres.data.download_MP2RAGEME_testdata(in_dir)

############################################################################
# Skull stripping
# ----------------
# First we perform skull stripping. Only the second inversion image is required
# to calculate the brain mask. But if we input the T1map and T1w image as well,
# they will be masked for us. We also save the outputs in the ``out_dir``
# specified above and use a subject ID as the base file_name.
skullstripping_results1 = nighres.brain.intensity_based_skullstripping(
                                            main_image=dataset['mp2rageme_inv2e1'],
                                            extra_image=dataset['mp2rageme_qt1'],
                                            save_data=True,
                                            output_dir=out_dir)

skullstripping_results2 = nighres.brain.intensity_based_skullstripping(
                                            main_image=dataset['slab_inv2e1'],
                                            extra_image=dataset['slab_qt1'],
                                            save_data=True,
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
    plotting.plot_roi(skullstripping_results1['brain_mask'], dataset['mp2rageme_qt1'],
                      annotate=False, black_bg=True, draw_cross=False,
                      cmap='autumn')
    plotting.plot_roi(skullstripping_results2['brain_mask'], dataset['slab_qt1'],
                      annotate=False, black_bg=True, draw_cross=False,
                      cmap='autumn')
############################################################################

#############################################################################
# SyN co-registration
# --------------------
# Next, we use the masked data as input for co-registration. The T1 maps are
# used here as they are supposed to be more similar
#
# Note: the SyN algorithm is fairly complex, and comes with a lot of options.
# the nighres wrapper comes with presets and additional helper features that
# aim to make the algorithm easier to work with, e.g. voxel-to-voxel mappings
# or automated mask boundary smoothing.

syn_results = nighres.registration.embedded_antspy(
                        source_image=skullstripping_results1['extra_masked'],
                        target_image=skullstripping_results2['extra_masked'],
                        run_rigid=True, run_affine=True, run_syn=True,
                        rigid_iterations=1000, affine_iterations=1000, 
                        coarse_iterations=40,
                        medium_iterations=20, fine_iterations=10,
                        cost_function='MutualInformation',
                        interpolation='Linear',
                        mask_zero=False, smooth_mask=0.1,
                        ignore_affine=True, ignore_header=False,
                        save_data=True, 
                        output_dir=out_dir)

#############################################################################
# Apply deformations to images
# ----------------------------
# Finally, we use the computed deformation to transform other associated images
deformed = nighres.registration.apply_coordinate_mappings(
                        image=dataset['mp2rageme_qt1'],
                        mapping1=syn_results['mapping'],
                        zero_border=1,
                        interpolation='linear',
                        save_data=True, 
                        output_dir=out_dir)

inverse = nighres.registration.apply_coordinate_mappings(
                        image=dataset['slab_qt1'],
                        mapping1=syn_results['inverse'],
                        zero_border=2,
                        interpolation='linear',
                        save_data=True, 
                        output_dir=out_dir)

############################################################################
# Now we look at the coregistered images from applying the deformation
if not skip_plots:
    plotting.plot_img(deformed['result'],
                      annotate=False,  black_bg=False, draw_cross=False)
    plotting.plot_img(inverse['result'],
                      annotate=False,  black_bg=False, draw_cross=False)

############################################################################

#############################################################################
# If the example is not run in a jupyter notebook, render the plots:
if not skip_plots:
    plotting.show()

#############################################################################
# References
# -----------
# .. [1] Alkemade, A., Mulder, M.J., Groot, J.M., Isaacs, B.R., van Berendonk, N., Lute, N., 
#        Isherwood, S.J., Bazin, P.-L., Forstmann, B.U., 2020. The Amsterdam Ultra-high field 
#        adult lifespan database (AHEAD): A freely available multimodal 7 Tesla submillimeter 
#        magnetic resonance imaging database. NeuroImage 221, 117200. 
#        https://doi.org/10.1016/j.neuroimage.2020.117200
# .. [2] Bazin, P.-L., Weiss, M., Dinse, J., Schäfer, A., Trampel, R., Turner, R., 2014. 
#        A computational framework for ultra-high resolution cortical segmentation at 7Tesla. 
#        NeuroImage 93, 201–209. https://doi.org/10.1016/j.neuroimage.2013.03.077
# .. [3] Avants et al (2008). Symmetric diffeomorphic image registration with
#        cross-correlation: evaluating automated labeling of elderly and
#        neurodegenerative brain. DOI: 10.1016/j.media.2007.06.004
