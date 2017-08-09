"""
Sampling T1 on equivolumetric layers from MP2RAGE data
=======================================================

Complete pipeline performing all steps necessary to use MP2RAGE data for
sampling T1 at different intracortical depth levels which are determined using
a volume-preserving approach:

* Dowloading public MP2RAGE data [1] :func:`nighres.download_from_url`
* Skull stripping :func:`nighres.mp2rage_skullstripping`
* Tissue segmentation using MGDM [2] :func:`nighres.mgdm_segmentation`
* Creating levelset representations of the pial and white matter surface
* Equivolumetric layering of the cortical sheet [3]
* Sampling T1 on the different intracortical depth

References
----------
[1] Tardif et al (2016). Open Science CBS Neuroimaging Repository: Sharing
ultra-high-field MR images of the brain. DOI: 10.1016/j.neuroimage.2015.08.042
[2] Bogovic, Prince and Bazin (2013). A multiple object geometric
deformable model for image segmentation. DOI: 10.1016/j.cviu.2012.10.006.A
[3] Waehnert et al (2014). Anatomically motivated modeling of cortical
laminae. DOI: 10.1016/j.neuroimage.2013.03.078
"""

import os
import urllib
import numpy as np
import nibabel as nb
import nighres

# Setting output directory and names for downloaded files
out_dir = '/SCR/data/nighres_testing/'
t1map = os.path.join(out_dir, "T1map.nii.gz")
t1w = os.path.join(out_dir, "T1w.nii.gz")

# Download example MP2RAGE data from the CBS Open Science repository
nighres.download_from_url("http://openscience.cbs.mpg.de/bazin/7T_Quantitative/MP2RAGE-05mm/subject01_mp2rage_0p5iso_qT1.nii.gz", t1map)  # noqa
nighres.download_from_url("http://openscience.cbs.mpg.de/bazin/7T_Quantitative/MP2RAGE-05mm/subject01_mp2rage_0p5iso_uni.nii.gz", t1w)  # noqa

# TODO: download INV2 from online
data_dir = '/SCR/data/cbstools_testing/'
inv2 = data_dir + '7t_trt/test_nii/INV2.nii.gz'

# Skullstripping of MP2RAGE images
skullstripping_results = nighres.mp2rage_skullstripping(
                                                        second_inversion=inv2,
                                                        t1_weighted=t1w,
                                                        t1_map=t1map,
                                                        save_data=True,
                                                        output_dir=out_dir)

# MGDM segmentation using the skullstripped T1 weighted and T1map image
segmentation_results = nighres.mgdm_segmentation(
                        contrast_image1=skullstripping_results.t1w_masked,
                        contrast_type1="Mp2rage7T",
                        contrast_image2=skullstripping_results.t1map_masked,
                        contrast_type2="T1map7T",
                        save_data=True, output_dir=out_dir)


# Creating binary representations of the pial and white matter surface
wm = [11, 12, 13, 17, 18, 30, 31, 32, 33, 34, 35, 36, 37,
      38, 39, 40, 41, 47, 48]
gm = [26, 27]

segmentation = segmentation_results['segmentation'].get_data()
wm_mask = np.zeros(segmentation.shape)
for x in wm:
    wm_mask[np.where(segmentation == x)] = 1
wm_nii = nb.Nifti1Image(wm_mask,
                        segmentation_results['segmentation'].get_affine())

gm_mask = np.copy(wm_mask)
for x in gm:
    gm_mask[np.where(segmentation == x)] = 1
gm_nii = nb.Nifti1Image(gm_mask,
                        segmentation_results['segmentation'].get_affine())

# Creating levelsets from binary tissue images
gm_wm_levelset = nighres.create_levelsets(probability_image=wm_nii,
                                          save_data=True)
gm_csf_levelset = nighres.create_levelsets(probability_image=gm_nii,
                                           save_data=True)

# Perform volumetric layering of the cortical sheet
layering_results = nighres.volumetric_layering(inner_levelset=gm_wm_levelset,
                                               outer_levelset=gm_csf_levelset,
                                               n_layers=5, save_data=True)

# Sample T1 across the different cortical depth levels
profiles = nighres.profile_sampling(
                        profile_surface_image=layering_results['boundaries'],
                        intensity_image=t1map, save_data=True)

# TODO: Visualize data using Nilearn
