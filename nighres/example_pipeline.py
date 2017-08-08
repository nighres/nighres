import os
import urllib
import numpy as np
import nibabel as nb
import nighres
from utils import _download_from_url

out_dir = '/SCR/data/nighres_testing/'
t1map = os.path.join(out_dir, "T1map.nii.gz")
t1w = os.path.join(out_dir, "T1w.nii.gz")
# TODO: where to get the INV2 file?

_download_from_url("http://openscience.cbs.mpg.de/bazin/7T_Quantitative/MP2RAGE-05mm/subject01_mp2rage_0p5iso_qT1.nii.gz", t1map)  # noqa
_download_from_url("http://openscience.cbs.mpg.de/bazin/7T_Quantitative/MP2RAGE-05mm/subject01_mp2rage_0p5iso_uni.nii.gz", t1w)  # noqa

data_dir = '/SCR/data/cbstools_testing/'
inv2 = data_dir + '7t_trt/test_nii/INV2.nii.gz'

# Skullstripping of MP2RAGE images
skullstripping_results = nighres.mp2rage_skullstripping(
                                                        second_inversion=inv2,
                                                        t1_weighted=t1w,
                                                        t1_map=t1map,
                                                        save_data=True,
                                                        output_dir=out_dir)

# MGDM segmentation using both the T1 weighted and image
segmentation_results = nighres.mgdm_segmentation(
                        contrast_image1=skullstripping_results.t1w_masked,
                        contrast_type1="Mp2rage7T",
                        contrast_image2=skullstripping_results.t1map_masked,
                        contrast_type2="T1map7T",
                        save_data=True, output_dir=out_dir)


# Creating binary representations of the GM/WM and GM/CSF boundaries from the
# segmentation output
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
gm_wm_levelset = nighres.create_levelsets(wm_mask, save_data=True)
gm_csf_levelset = nighres.create_levelsets(gm_mask, save_data=True)

# Perform volumetric layering of the cortical sheet
depth, layers, boundaries = nighres.layering(gm_wm_levelset,
                                             gm_csf_levelset,
                                             n_layers=3,
                                             save_data=False)

# sample t1 across cortical layers
profiles = nighres.profile_sampling(boundaries,
                                    t1map,
                                    save_data=False)
