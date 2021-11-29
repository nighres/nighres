# Plots are currently included as images, because example is too big to
# run on readthedocs servers
"""
Subcortex Parcellation
======================

This example shows how to perform multi-contrast subcortical parcellation
with the MASSP algorithm on MP2RAGEME data by performing the following steps:

1. Downloading an open MP2RAGE datasets using
    :func:`nighres.data.download_MP2RAGEME_sample` [1]_
2. Downloading the open AHEAD template using
    :func:`nighres.data.download_AHEAD_template` [2]_
3. Register the data to the AHEAD brain template
    :func:`nighres.registration.embedded_antspy` [3]_
4. Subcortex parcellation with MASSP
   :func:`nighres.parcellation.massp` [4]_

Note: MASSP labels are listed inside the corresponding module and can be
accessed with :func:`nighres.parcellation.massp_17structures_label`
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
out_dir = os.path.join(os.getcwd(), 'nighres_examples/massp_parcellation')

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
# Now we download an example MP2RAGEME dataset, including a quantitative R1 map,
# a quantitative R2* map, and a QSM, all skull-stripped.
dataset = nighres.data.download_MP2RAGEME_sample(data_dir=in_dir)
############################################################################
# Now we download the AHEAD template for coregistration to the atlas space.
template = nighres.data.download_AHEAD_template()
############################################################################
# Co-registration
# First we co-register the subject to the AHEAD template, and save the 
# transformation mappings in the ``out_dir`` specified above, using a subject 
# ID as the base file_name.
# Note that we register the template to the subject here, as the subject space is smaller
# (to limit memory usage)
ants1 = nighres.registration.embedded_antspy_multi(
                        source_images=[template['qr1'],template['qr2s'],template['qsm']],
                        target_images=[dataset['qr1'],dataset['qr2s'],dataset['qsm']],
                        run_rigid=True, run_affine=True, run_syn=True,
                        rigid_iterations=1000,
                        affine_iterations=500,
                        coarse_iterations=180, 
                        medium_iterations=60, fine_iterations=30,
                        cost_function='MutualInformation', 
                        interpolation='NearestNeighbor',
                        regularization='High',
                        ignore_affine=True, 
                        save_data=True, file_name="mni2subject-step1",
                        output_dir=out_dir)

# Co-registration to an atlas works better in two steps
ants2 = nighres.registration.embedded_antspy_multi(
                        source_images=[ants1['transformed_sources'][0],
                                       ants1['transformed_sources'][1],
                                       ants1['transformed_sources'][2]],
                        target_images=[dataset['qr1'],dataset['qr2s'],dataset['qsm']],
                        run_rigid=True, run_affine=True, run_syn=True,
                        rigid_iterations=1000,
                        affine_iterations=500,
                        coarse_iterations=180, 
                        medium_iterations=60, fine_iterations=30,
                        cost_function='MutualInformation', 
                        interpolation='NearestNeighbor',
                        regularization='High',
                        ignore_affine=True, 
                        save_data=True, file_name="mni2subject-step2",
                        output_dir=out_dir)

# combine transformations
mapping = nighres.registration.apply_coordinate_mappings(
                        image=ants1['mapping'],
                        mapping1=ants2['mapping'],
                        interpolation='linear',
                        save_data=True, file_name="mapping2subject",
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
    plotting.plot_anat(ants2['transformed_source'],cut_coords=[0.0,0.0,0.0],
                      annotate=False,  draw_cross=False)
    plotting.plot_anat(template['qr1'],cut_coords=[0.0,0.0,0.0],
                      annotate=False,  draw_cross=False)
    plotting.show()
############################################################################


#############################################################################
# MASSP Parcellation
# ---------------------
# Finally, we use the MASSP algorithm to parcellate the subcortex
massp = nighres.parcellation.massp(target_images=[dataset['qr1'],dataset['qr2s'],dataset['qsm']],
                                map_to_target=mapping['result'], 
                                max_iterations=120, max_difference=0.1,
                                save_data=True, file_name="sample-subject",
                                output_dir=out_dir, overwrite=False)


############################################################################
# Now we look at the topology-constrained segmentation MGDM created
if not skip_plots:
    plotting.plot_roi(massp['max_label'], dataset['qr1'],
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='cubehelix')
    plotting.plot_img(massp['max_proba'],
                      vmin=0, vmax=1, cmap='gray',  colorbar=True,
                      annotate=False,  draw_cross=False)

############################################################################


#############################################################################
# If the example is not run in a jupyter notebook, render the plots:
if not skip_plots:
    plotting.show()

#############################################################################
# References
# -----------
# .. [1] Caan et al. (2018) MP2RAGEME: T1, T2*, and QSM mapping in one sequence 
#    at 7 tesla. DOI: 10.1002/hbm.24490
# .. [2] Alkemade et al (under review). The Amsterdam Ultra-high field adult 
#    lifespan database (AHEAD): A freely available multimodal 7 Tesla 
#    submillimeter magnetic resonance imaging database.
# .. [3] Avants et al (2008). Symmetric diffeomorphic image registration with
#    cross-correlation: evaluating automated labeling of elderly and
#    neurodegenerative brain. DOI: 10.1016/j.media.2007.06.004
# .. [4] Bazin et al. (in prep) Multi-contrast Anatomical Subcortical 
#    Structures Parcellation
