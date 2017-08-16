"""
Sampling T1 at different intracortical depths
==============================================

This example shows a complete pipeline performing all steps necessary to use
MP2RAGE data for sampling T1 at different intracortical depth levels,
determined using a volume-preserving approach :

* Dowloading openly available MP2RAGE data [1]_
* Skull stripping :func:`nighres.brain.mp2rage_skullstripping`
* Tissue segmentation using MGDM :func:`nighres.brain.mgdm_segmentation` [2]_
* Creating levelset representations of the pial and white matter surface
  :func:`nighres.surface.probability_to_levelset`
* Equivolumetric layering of the cortical sheet
  :func:`nighres.laminar.volumetric_layering` [3]_
* Sampling T1 on the different intracortical depth
  :func:`nighres.laminar.profile_sampling`
"""

############################################################################
# Import and download
# -------------------
# First we import nighres and the os module to set the output directory and
# filenames for the files we will download. Make sure to run this file in a
# directory you have write-access to, or change the out_dir variable below.

import nighres
import os

out_dir = os.path.join(os.getcwd(), 'nighres_cache/t1_sampling')
# t1map = os.path.join(out_dir, "T1map.nii.gz")
# t1w = os.path.join(out_dir, "T1w.nii.gz")

############################################################################
# Now we download an example MP2RAGE dataset from the
# CBS Open Science repository
#
# .. todo:: also download INV2, for now using local version
inv2 = '/home/pilou/Projects/Nighres-testing-Julia/testdata/INV2.nii.gz'
t1map = '/home/pilou/Projects/Nighres-testing-Julia/testdata/T1map.nii.gz'
t1w = '/home/pilou/Projects/Nighres-testing-Julia/testdata/T1w.nii.gz'

# nighres.download_from_url("http://openscience.cbs.mpg.de/bazin/7T_Quantitative/MP2RAGE-05mm/subject01_mp2rage_0p5iso_qT1.nii.gz",
#                           t1map)
# nighres.download_from_url("http://openscience.cbs.mpg.de/bazin/7T_Quantitative/MP2RAGE-05mm/subject01_mp2rage_0p5iso_uni.nii.gz",
#                           t1w)
############################################################################
# Tissue classification
# ---------------------
# The first processing step is to skullstrip the images. Only the second
# inversion image is required to calculate the brain mask. But if we input
# the T1map and T1w image as well, they will be masked for us.
skullstripping_results = nighres.brain.mp2rage_skullstripping(
                                                        second_inversion=inv2,
                                                        t1_weighted=t1w,
                                                        t1_map=t1map,
                                                        save_data=True,
                                                        output_dir=out_dir)

############################################################################
# .. tip:: in Nighres, functions that have several outputs return a
#    dictionary storing the different outputs. You can find the keys in the
#    docstring or list them like this:

skullstripping_results.keys()

############################################################################
# Next we use the masked data as input for tissue segmentation with the MGDM
# algorithm. The segmentation works with a single contrast, but can  be
# improved with additional contrasts. In this case we use the T1-weigthed
# image as well as the quantitative T1map.
segmentation_results = nighres.brain.mgdm_segmentation(
                        contrast_image1=skullstripping_results['t1w_masked'],
                        contrast_type1="Mp2rage7T",
                        contrast_image2=skullstripping_results['t1map_masked'],
                        contrast_type2="T1map7T",
                        save_data=True, output_dir=out_dir)

############################################################################
# Creating surfaces
# -----------------
# To create levelset representations of the pial and white matter surface,
# we first use the segmentation results to create binary masks representing
# those boundaries. This small thresholding loop bypasses a step that would
# usually be performed with CRUISE, an interface that has not yet been
# included in Nighres
#
# In the atlas file we used for MGDM segmentation, we can find the labels
# corresponding to subcortical and white matter (wm) structures and cortical
# gray matter (gm). We used the default atlas file, its location is stored
# in the DEFAULT_ATLAS variable. You can take a look at it yourself to see
# where the numbers in the following lists come frome

with open(nighres.DEFAULT_ATLAS, 'r') as f:
    print(f.read())

wm = [11, 12, 13, 17, 18, 30, 31, 32, 33, 34, 35, 36, 37,
      38, 39, 40, 41, 47, 48]
gm = [26, 27]

############################################################################
# .. tip:: Since data is passed as Nibabel objects, we can manipulate it
#    directly in Python using Numpy and Nibabel

import numpy as np
import nibabel as nb

segmentation = segmentation_results['segmentation'].get_data()
affine = segmentation_results['segmentation'].get_affine()
header = segmentation_results['segmentation'].get_header()

wm_mask = np.zeros(segmentation.shape)
for x in wm:
    wm_mask[np.where(segmentation == x)] = 1

gm_mask = np.copy(wm_mask)
for x in gm:
    gm_mask[np.where(segmentation == x)] = 1

# Make Nifti objects for further processing and saving
# Adapting header max value for display
header['cal_max'] = 1
wm_nii = nb.Nifti1Image(wm_mask, affine, header)
wm_nii.to_filename(os.path.join(out_dir, 'wm_mask.nii.gz'))
gm_nii = nb.Nifti1Image(gm_mask, affine, header)
gm_nii.to_filename(os.path.join(out_dir, 'pial_mask.nii.gz'))


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

###########################################################################
# Creating layers and sampling
# ----------------------------
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
# .. [1] Tardif et al (2016). Open Science CBS Neuroimaging Repository: Sharing
#    ultra-high-field MR images of the brain.
#    DOI: 10.1016/j.neuroimage.2015.08.042
# .. [2] Bogovic, Prince and Bazin (2013). A multiple object geometric
#    deformable model for image segmentation. DOI: 10.1016/j.cviu.2012.10.006.A
# .. [3] Waehnert et al (2014). Anatomically motivated modeling of cortical
#    laminae. DOI: 10.1016/j.neuroimage.2013.03.078
