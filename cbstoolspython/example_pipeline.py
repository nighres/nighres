import cbstoolspython
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn._utils.niimg_conversions import _index_img


out_dir = '/tmp'
inv2 = '/SCR/data/7t_trt/test_nii/INV2.nii.gz'
t1w = '/SCR/data/7t_trt/test_nii/T1w.nii.gz'
t1map = '/SCR/data/7t_trt/test_nii/T1map.nii.gz'

skullstripping_results = cbstoolspython.mp2rage_skullstripping(
                                                        second_inversion=inv2,
                                                        t1_weighted=t1w,
                                                        t1_map=t1map,
                                                        save_data=True,
                                                        output_dir=out_dir)

segmentation_results = cbstoolspython.mgdm_segmentation(
                        contrast_image1=skullstripping_results.t1w_masked,
                        contrast_type1="Mp2rage7T",
                        contrast_image2=skullstripping_results.t1map_masked,
                        contrast_type2="Mp2rage7T",
                        save_data=True, output_dir=out_dir)

# get gm / wm tissue class images
#
# gm_wm_levelset = cbstoolspython.create_levelsets(gwb_prob, save_data=True)
# gm_csf_levelset = cbstoolspython.create_levelsets(cgb_prob, save_data=True)
#
# depth, layers, boundaries = cbstoolspython.layering(gwb, cgb,
#                                                     n_layers=n_layers,
#                                                     save_data=False)
#
# profiles = cbstoolspython.profile_sampling(boundaries, intensity,
#                                            save_data=False)
#n_layers = 3
#coords = (-5, -2, 1)
#gwb_prob = './data/adult_F04_intern_orig_binmask.nii.gz'
#cgb_prob = './data/adult_F04_extern_orig_binmask.nii.gz'
#intensity = './data/F04_01032013_MSME_TEsum_magn_initial.nii'
#mesh = './data/adult_F04_midcortical_surf.vtk'
# fig1 = plt.figure(figsize=(20, 8))
# ax1 = fig1.add_subplot(211)
# plotting.plot_anat(gwb_prob, annotate=False, draw_cross=False,
#                    display_mode='z', cut_coords=coords,
#                    figure=fig1, axes=ax1, title='GM/WM boundary', vmax=1)
# ax2 = fig1.add_subplot(212)
# plotting.plot_anat(cgb_prob, annotate=False, draw_cross=False,
#                    display_mode='z', cut_coords=coords,
#                    figure=fig1, axes=ax2, title='GM/CSF boundary', vmax=1)
#
# fig2 = plt.figure(figsize=(20, 8))
# ax1 = fig2.add_subplot(211)
# plotting.plot_anat(gwb, annotate=False, draw_cross=False,
#                    display_mode='z', cut_coords=coords,
#                    figure=fig2, axes=ax1, title='GM/WM boundary',
#                    vmin=gwb.get_data().min())
# ax2 = fig2.add_subplot(212)
# plotting.plot_anat(cgb, annotate=False, draw_cross=False,
#                    display_mode='z', cut_coords=coords,
#                    figure=fig2, axes=ax2, title='GM/CSF boundary',
#                    vmin=cgb.get_data().min())
#
#
# fig3 = plt.figure(figsize=(20, 8))
# ax1 = fig3.add_subplot(211)
# plotting.plot_img(depth, annotate=False, draw_cross=False,
#                   display_mode='z', cut_coords=coords,
#                   figure=fig3, axes=ax1, title='Equivolumetric depth',
#                   cmap='inferno')
# ax2 = fig3.add_subplot(212)
# plotting.plot_img(layers, annotate=False, draw_cross=False, display_mode='z',
#                   cut_coords=coords, figure=fig3, axes=ax2,
#                   title='Equivolumetric layers', cmap='gnuplot',
#                   vmin=0, vmax=5)
#
#
# fig4 = plt.figure(figsize=(20, (n_layers + 1) * 4))
# for i in range(n_layers + 1):
#     ax = fig4.add_subplot(n_layers + 1, 1, i + 1)
#     plotting.plot_img(_index_img(profiles, i), annotate=False,
#                       draw_cross=False, display_mode='z',
#                       cut_coords=coords, figure=fig4, axes=ax,
#                       title='Depth %s' % str(i), cmap='inferno',
#                       vmin=0, vmax=0.6 * 1e8)
# plotting.show()
