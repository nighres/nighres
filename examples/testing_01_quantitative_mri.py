"""
Quantitative MRI
=======================================

This example shows how to build quantitative maps of R1 and R2* and semi-quantitative
PD from a MP2RAGEME dataset by performing the following steps:

1. Download a downsampled MP2RAGEME dataset using 
    :func:`nighres.data.download_MP2RAGEME_testdata` [1]_
2. Denoise the MP2RAGEME data using
   :func:`nighres.intensity.lcpca_denoising` [2]_
3. Build quantitative maps using
   :func:`nighres.intensity.mp2rage_t1_mapping`, 
   :func:`nighres.intensity.flash_t2s_fitting`, 
   :func:`nighres.intensity.mp2rageme_pd_mapping` [3]_
4. Remove the skull and create a brain mask using
   :func:`nighres.brain.mp2rage_skullstripping` [4]_
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
out_dir = os.path.join(os.getcwd(), 'nighres_testing/quantitative_mri')

############################################################################
# We import the ``ants`` modules for inhomogeneity correction
# As ANTspy is a depencency for nighres, it should be installed already

import ants

############################################################################
# We also import the ``numpy`` and ``nibabel`` modules to perform basic image
# operations like masking, intensity scaling, or reorientation

import nibabel
import numpy

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
dataset = nighres.data.download_MP2RAGEME_testdata(in_dir)

############################################################################
# Denoising
# ----------------
# First we perform some denoising. Quantitative MRI sequences combine multiple
# acquisitions (in this case 10, counting magnitude and phase) which can be efficiently
# denoised with a local PCA approach without compromising spatial resolution [2]_.
denoising_results = nighres.intensity.lcpca_denoising(
                                      image_list=dataset['mp2rageme_mag'], 
                                      phase_list=dataset['mp2rageme_phs'], 
                                      unwrap=True, rescale_phs=True,
                                      save_data=True, 
                                      output_dir=out_dir)

############################################################################
# .. tip:: in Nighres functions that have several outputs return a
#    dictionary storing the different outputs. You can find the keys in the
#    docstring by typing ``nighres.intensity.lcpca_denoising?`` or list
#    them with ``denoising_results.keys()``
#
# .. tip: in Nighres modules check whether computations have been run or 
#    not, and skip them if the output files exist. You can force overwriting 
#    with the input option ``overwrite=True``
#
# .. tip: file names given to modules serve as base names for the output
#    unless a specific name in provided by ``file_name=`` or ``file_names=[]``
#    and the module always adds suffixes related to its outputs.
#
# To check if the denoising worked well we plot one of the original images
# and the corresponding denoised result. You can also open the images stored 
# in ``out_dir`` in your favourite interactive viewer and scroll through the volume.
#
# Like Nilearn, we use Nibabel SpatialImage objects to pass data internally.
# Therefore, we can directly plot the outputs using `Nilearn plotting functions
# <http://nilearn.github.io/plotting/index.html#different-plotting-functions>`_
# .

if not skip_plots:
    plotting.plot_anat(dataset['inv1m'], cmap='gray')
    plotting.plot_anat(denoising_results['denoised'][0], cmap='gray')



############################################################################
# T1 quantitative mapping
# ----------------
# Now we can generate a T1 and R1 map (with R1=1/T1). 
# Note that we could skip the denoising and use the original data directly. 
# The T1 mapping requires several of the imaging parameters, which are often 
# available from headers and/or json files generated when exporting images 
# from the scanner into a standard format like Nifti. 
# By default Nighres does not extract them automatically, they have to be
# explicitly provided.
#
# Note also that quantitative T1, T2* parameters have units: here we use
# seconds (and Hertz) as the basis. Sometimes people prefer milliseconds,
# so a x1000 scaling factor is expected.

# T1 mapping uses the first and second inversion, first echo, both magnitude and phase
inv1m = denoising_results['denoised'][0]
inv1p = denoising_results['denoised'][5]

inv2e1m = denoising_results['denoised'][1]
inv2e1p = denoising_results['denoised'][6]

t1mapping_results = nighres.intensity.mp2rage_t1_mapping(
                                                 first_inversion=[inv1m,inv1p],
                                                 second_inversion=[inv2e1m,inv2e1p],
                                                 excitation_TR=[0.0062,0.0314],
                                                 flip_angles=[7.0,6.0],
                                                 inversion_TR=6.720,
                                                 inversion_times=[0.607,3.855],
                                                 N_excitations=150,
                                                 save_data=True,
                                                 output_dir=out_dir)

if not skip_plots:
    plotting.plot_anat(t1mapping_results['r1'], vmax=3.0, cmap='gray')
    plotting.plot_anat(t1mapping_results['t1'], vmax=4.0, cmap='gray')

############################################################################
# Quantitative T2* fitting
# ----------------
# The relevant images for T2* fitting are the 4 echoes from the second inversion.
# As with T1 mapping, echo times (TE) are explicitly provided

inv2e2m = denoising_results['denoised'][2]
inv2e3m = denoising_results['denoised'][3]
inv2e4m = denoising_results['denoised'][4]

t2sfitting_results = nighres.intensity.flash_t2s_fitting(
                                         image_list=[inv2e1m,inv2e2m,inv2e3m,inv2e4m],
                                         te_list=[0.0030,0.0115,0.0200,0.0285],
                                         save_data=True,
                                         output_dir=out_dir)

if not skip_plots:
    plotting.plot_anat(t2sfitting_results['r2s'], vmax=100.0 cmap='gray')
    plotting.plot_anat(t2sfitting_results['t2s'], vmax=0.1, cmap='gray')

############################################################################
# Semi-quantitative PD mapping
# ----------------
# PD mapping combines information from T1 and T2* mapping with the data from
# the first and second inversion. PD estimates are not normalized to a specific
# region value (e.g. ventricles, white matter...)

pdmapping_results = nighres.intensity.mp2rageme_pd_mapping(
                                        first_inversion=[inv1m,inv1p]                                  
                                        second_inversion=[inv2e1m,inv2e1p],
                                        t1map=t1mapping_results['t1'], 
                                        r2smap=t2sfitting_results['r2s'],
                                        echo_times=[0.0030,0.0115,0.0200,0.0285],
                                        inversion_times=[0.670, 3.85],
                                        flip_angles=[4.0, 4.0],
                                        inversion_TR=6.72,
                                        excitation_TR=[0.0062, 0.0314],
                                        N_excitations=150,
                                        save_data=True,
                                        output_dir=out_dir)

if not skip_plots:
    plotting.plot_anat(pdmapping_results['pd'], cmap='gray')


############################################################################
# Quantitative susceptibility mapping (QSM)
# ----------------
# Note that this data can also be used to obtain QSM, using the phase data
# from the second inversion. Nighres does not include a QSM reconstruction
# technique, but we have used successfully TGV-QSM, which has the advantage
# of being a python-based software tool (other methods may be superior, but
# few run as standalone or python scripts).


############################################################################
# Skull stripping
# ----------------
# Finally, we perform skull stripping, and apply it to all the quantitative maps. 
# Only the second inversion, first echo image is required to calculate the brain mask. 
# But if we input the T1map and/or T1w image as well, they will help refine the CSF
# boundary.
skullstripping_results = nighres.brain.mp2rage_skullstripping(
                                            second_inversion=inv2e1m,
                                            t1_weighted=t1mapping_results['uni'],
                                            t1_map=t1mapping_results['t1'],
                                            save_data=True,
                                            output_dir=out_dir)

if not skip_plots:
    plotting.plot_roi(skullstripping_results['brain_mask'], t1mapping_results['r1'],
                      annotate=False, black_bg=False, draw_cross=False,
                      cmap='gray')

############################################################################
# Masking, Thresholding, and Reorientation
# ----------------
# Here we use nibabel and numpy routines to perform these simple steps,
# rather than having a dedicated nighres module
# Note that thresholds have been set for 7T qMRI values, and would need
# to be updated for other field strengths.
# Note also that the PD map is normalized to the mean, after inhomogeneity correction

brainmask_file = skullstripping_results['brain_mask']
brainmask = nighres.io.load_volume(brainmask_file).get_fdata()

r1strip_file = t1mapping_results.replace('.nii','_brain.nii')
if not os.path.isfile(r1strip_file):
    print("Mask qR1")
    r1 = nighres.io.load_volume(t1mapping_results['r1'])
    r1strip = nibabel.Nifti1Image(numpy.minimum(3.0,brainmask*r1.get_fdata()), r1.affine, r1.header)
    r1strip = nibabel.as_closest_canonical(r1strip)
    nighres.io.save_volume(r1strip_file, r1strip)
    
r2strip_file = t2sfitting_results['r2s'].replace('.nii','_brain.nii')
if not os.path.isfile(r2strip_file):
    print("Mask qR2s")
    r2 = nighres.io.load_volume(t2sfitting_results['r2s'])
    r2strip = nibabel.Nifti1Image(numpy.maximum(0.0,numpy.minimum(200.0,brainmask*r2.get_fdata())), r2.affine, r2.header)
    r2strip = nibabel.as_closest_canonical(r2strip)
    nighres.io.save_volume(r2strip_file, r2strip)t2s_img = nighres.io.load_volume(t2smapping['t2s'])

# for PD, we also need to run some inhomogeneity correction with N4
pdn4_file = pdmapping_results['pd'].replace('.nii','_n4.nii')
if not os.path.isfile(pdn4_file):
    print("Correct inhomogeneities for PD")
    img = ants.image_read(pdmapping_results['pd'])
    pd_n4 = ants.n4_bias_field_correction(img, mask=brainmask_file)
    ants.image_write(pd_n4, pdn4_file)
   
pdstrip_file = pdn4_file.replace('.nii','_brain.nii')
if not os.path.isfile(pdstrip_file):
    print("Mask PD")
    pd = nighres.io.load_volume(pdn4_file)
    pddata = pd.get_fdata()
    pdmean = numpy.mean(pddata[pddata>0])
    pdstrip = nibabel.Nifti1Image(numpy.minimum(8.0,brainmask*pddata/pdmean), pd.affine, pd.header)
    pdstrip = nibabel.as_closest_canonical(pdstrip)
    nighres.io.save_volume(pdstrip_file, pdstrip)

############################################################################
# And we are done! Let's have a look at the final maps:

if not skip_plots:
    plotting.plot_anat(r1strip_file, cmap='gray')
    plotting.plot_anat(r2strip_file, cmap='gray')
    plotting.plot_anat(pdstrip_file, cmap='gray')
    

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
# .. [2] Bazin, P.-L., Alkemade, A., van der Zwaag, W., Caan, M., Mulder, M., Forstmann, 
#        B.U., 2019. Denoising High-Field Multi-Dimensional MRI With Local Complex PCA. 
#        Frontiers in Neuroscience 13. https://doi.org/10.3389/fnins.2019.01066
# .. [3] Caan, M.W.A., Bazin, P., Marques, J.P., Hollander, G., Dumoulin, S.O., Zwaag, W., 2019. 
#        MP2RAGEME: T1, T2*, and QSM mapping in one sequence at 7 tesla. 
#        Human Brain Mapping 40, 1786–1798. https://doi.org/10.1002/hbm.24490
# .. [4] Bazin, P.-L., Weiss, M., Dinse, J., Schäfer, A., Trampel, R., Turner, R., 2014. 
#        A computational framework for ultra-high resolution cortical segmentation at 7Tesla. 
#        NeuroImage 93, 201–209. https://doi.org/10.1016/j.neuroimage.2013.03.077
