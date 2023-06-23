# Plots are currently included as images, because example is too big to
# run on readthedocs servers
"""
Subcortex Parcellation
======================

This example shows how to perform multi-contrast subcortical parcellation
with the MASSP algorithm on MP2RAGEME data by performing the following steps:

1. Downloading an open MP2RAGE datasets using
    :func:`nighres.data.download_MP2RAGEME_testdata` [1]_
2. Downloading the open AHEAD template using
    :func:`nighres.data.download_AHEAD_template` [2]_
3. Register the data to the AHEAD brain template
    :func:`nighres.registration.embedded_antspy` [3]_
4. Subcortex parcellation with MASSP
   :func:`nighres.parcellation.massp` [4]_

Note: MASSP labels are listed inside the corresponding module and can be
accessed with :func:`nighres.parcellation.massp_17structures_label`

Note: MASSP requires multiple contrasts to properly identify all the structures
in its atlas, with both myelin and iron weighting (ideally R1 and R2*, with
possibly QSM and PDw if available). 
The standard atlas currently expects R1, R2*, QSM (in that order).

Note: MASSP works in 0.5mm MNI space: even for low resolution data, this 
means large atlas files will be downloaded and some of the processing might
require more memory than available on a typical laptop.
"""

############################################################################
# Import and download
# -------------------
# First we import ``nighres`` and the ``os`` module to set the output directory
# Make sure to run this file in a  directory you have write access to, or
# change the ``out_dir`` variable below.

import nighres
import os
import glob

in_dir = os.path.join(os.getcwd(), 'nighres_testing/quantitative_mri')
out_dir = os.path.join(os.getcwd(), 'nighres_testing/massp_parcellation')
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
# Now we import data from the first testing pipeline, which processed a MP2RAGEME dataset, 
# to generate quantitative R1, R2* and PD-weighted images, reoriented and skullstripped.
qr1_file = glob.glob(in_dir+'/*r1_brain.nii*')
qr2s_file = glob.glob(in_dir+'/*r2s_brain.nii*')
if len(qr1_file)*len(qr2s_file)==0:
    print("input files not found: did you run the 'testing_01_quantitative_mri.py' script?")
    exit()
else:
    qr1_file = qr1_file[0]
    qr2s_file = qr2s_file[0]
    dataset = {'qr1': qr1_file, 'qr2s':qr2s_file}
 
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
                        source_images=[template['qr1'],template['qr2s']],
                        target_images=[dataset['qr1'],dataset['qr2s']],
                        run_rigid=True, run_affine=True, run_syn=True,
                        rigid_iterations=1000,
                        affine_iterations=500,
                        coarse_iterations=180, 
                        medium_iterations=60, fine_iterations=30,
                        cost_function='MutualInformation', 
                        interpolation='NearestNeighbor',
                        regularization='High',
                        ignore_affine=True, 
                        save_data=True,
                        output_dir=out_dir)

# Co-registration to an atlas works better in two steps with ANTs
ants2 = nighres.registration.embedded_antspy_multi(
                        source_images=[ants1['transformed_sources'][0],
                                       ants1['transformed_sources'][1]],
                        target_images=[dataset['qr1'],dataset['qr2s']],
                        run_rigid=True, run_affine=True, run_syn=True,
                        rigid_iterations=1000,
                        affine_iterations=500,
                        coarse_iterations=180, 
                        medium_iterations=60, fine_iterations=30,
                        cost_function='MutualInformation', 
                        interpolation='NearestNeighbor',
                        regularization='High',
                        ignore_affine=True, 
                        save_data=True,
                        output_dir=out_dir)

# combine transformations
mapping = nighres.registration.apply_coordinate_mappings(
                        image=ants1['mapping'],
                        mapping1=ants2['mapping'],
                        interpolation='linear',
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
    plotting.plot_anat(ants2['transformed_sources'][0],cut_coords=[-75.0,90.0,-30.0],
                      annotate=False,  draw_cross=False)
    plotting.plot_anat(dataset['qr1'],cut_coords=[-75.0,90.0,-30.0], vmax=2.0,
                      annotate=False,  draw_cross=False)
############################################################################


#############################################################################
# MASSP Parcellation
# ---------------------
# Finally, we use the MASSP algorithm to parcellate the subcortex
massp = nighres.parcellation.massp(target_images=[dataset['qr1'],dataset['qr2s']],
                                map_to_target=mapping['result'], 
                                max_iterations=120, max_difference=0.1,
                                intensity_prior=0.01,
                                save_data=True, file_name="sample-subject",
                                output_dir=out_dir, overwrite=False)


############################################################################
# Now we look at the topology-constrained segmentation MGDM created
if not skip_plots:
    plotting.plot_roi(massp['max_label'], dataset['qr1'],cut_coords=[-75.0,90.0,-30.0],
                      annotate=False, black_bg=True, draw_cross=False,
                      cmap='cubehelix')
    plotting.plot_img(massp['max_proba'],cut_coords=[-75.0,90.0,-30.0],
                      vmin=0, vmax=1, cmap='gray',  colorbar=False,
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
