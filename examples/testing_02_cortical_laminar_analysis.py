"""
Cortical laminar analysis
=======================================

This example shows how to obtain a cortical laminar depth representation from
quantitative MRI data with the following steps:

1. Get a preprocessed and skullstripped R1 map result from the *quantitative_mri* 
    testing example
2. Define brain regions and tissues using MGDM
   :func:`nighres.brain.mgdm_segmentation` [1]_
3. Extract the cortex of the left hemisphere with
   :func:`nighres.brain.extract_brain_region`
4. Cortical reconstruction with CRUISE
   :func:`nighres.cortex.cruise_cortex_extraction` [2]_
5. Anatomical depth estimation through
   :func:`nighres.laminar.volumetric_layering` [3]_
6. Cortical surface inflation and mapping with
   :func:`nighres.surface.levelset_to_mesh`,
   :func:`nighres.surface.surface_inflation` [4]_
   and :func:`nighres.surface.surface_mesh_mapping`
   
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
out_dir = os.path.join(os.getcwd(), 'nighres_testing/cortical_laminar_analysis')
os.makedirs(out_dir, exist_ok=True)

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
# We also try to import Pyvista 3D rendering functions. If Pyvista is not
# installed, 3D rendering will be skipped.
skip_3d = False
try:
    import pyvista
except ImportError:
    skip_3d = True
    print('Pyvista could not be imported, 3D rendering will be skipped')


############################################################################
# Now we import data from the first testing pipeline, which processed a MP2RAGEME dataset, 
# to generate quantitative R1, R2* and PD-weighted images, reoriented and skullstripped.
dataset = glob.glob(in_dir+'/*r1_brain.nii*')
if len(dataset)==0:
    print("input file not found: did you run the 'testing_01_quantitative_mri.py' script?")
    exit()
else:
    dataset = dataset[0]
    
############################################################################
# MGDM classification
# ---------------------
# Next, we use the masked data as input for tissue classification with the MGDM
# algorithm. MGDM can work with a single T1-weighted contrast, but can  work with
# additional contrasts. Here we use a quantitative T1map, which we derive from
# the skull-stripped R1 map.

t1_file = dataset.replace(in_dir,out_dir).replace('.nii','_t1.nii')
if not os.path.isfile(t1_file):
    print("Build T1 map")
    r1_img = nighres.io.load_volume(dataset)
    mask = r1_img.get_fdata()>0.0001
    t1_data = r1_img.get_fdata()
    t1_data[mask] = 1.0/t1_data[mask]
    t1_img = nibabel.nifti1.Nifti1Image(t1_data,affine=r1_img.affine, header=r1_img.header)
    nighres.io.save_volume(t1_file,t1_img)

mgdm_results = nighres.brain.mgdm_segmentation(
                        contrast_image1=t1_file,
                        contrast_type1="T1map7T",
                        save_data=True,
                        output_dir=out_dir)

############################################################################
# Now we look at the topology-constrained segmentation MGDM created
if not skip_plots:
    plotting.plot_img(mgdm_results['segmentation'],cut_coords=[-75.0,90.0,-30.0],
                      vmin=1, vmax=50, cmap='cubehelix',  colorbar=True,
                      annotate=False,  draw_cross=False)

###########################################################################
# Region Extraction
# ------------------
# Here we pull from the MGDM output the needed regions for cortical
# reconstruction: the GM cortex ('region'), the underlying WM (with filled
# subcortex and ventricles, 'inside') and the surrounding CSF (with masked
# regions, 'background')
# Note there are multiple options for left, right or combined cerebrum
# and combined cerebellum.
cortex = nighres.brain.extract_brain_region(segmentation=mgdm_results['segmentation'],
                                            levelset_boundary=mgdm_results['distance'],
                                            maximum_membership=mgdm_results['memberships'],
                                            maximum_label=mgdm_results['labels'],
                                            extracted_region='left_cerebrum',
                                            save_data=True,
                                            output_dir=out_dir)

############################################################################
# To check if the extraction worked well we plot the GM and WM probabilities.
# You can also open the images stored in ``out_dir`` in
# your favourite interactive viewer and scroll through the volume.
#
if not skip_plots:
    plotting.plot_img(cortex['region_proba'],cut_coords=[-75.0,90.0,-30.0],
                      vmin=0, vmax=1, cmap='autumn',  colorbar=True,
                      annotate=False,  draw_cross=False)
    plotting.plot_img(cortex['inside_proba'],cut_coords=[-75.0,90.0,-30.0],
                      vmin=0, vmax=1, cmap='autumn',  colorbar=True,
                      annotate=False,  draw_cross=False)
############################################################################
    

#############################################################################
# CRUISE cortical reconstruction
# --------------------------------
# Next, we use the extracted data as input for cortex reconstruction with the
# CRUISE algorithm. CRUISE works with the membership functions as a guide and
# the WM inside mask as a (topologically spherical) starting point to grow a
# refined GM/WM boundary and CSF/GM boundary
cruise = nighres.cortex.cruise_cortex_extraction(
                        init_image=cortex['inside_mask'],
                        wm_image=cortex['inside_proba'],
                        gm_image=cortex['region_proba'],
                        csf_image=cortex['background_proba'],
                        normalize_probabilities=True,
                        save_data=True,
                        output_dir=out_dir)

###########################################################################
# Now we look at the topology-constrained segmentation CRUISE created
if not skip_plots:
    plotting.plot_img(cruise['cortex'],cut_coords=[-75.0,90.0,-30.0],
                      vmin=0, vmax=2, cmap='cubehelix',  colorbar=True,
                      annotate=False,  draw_cross=False)

###########################################################################


###########################################################################
# Volumetric layering
# ---------------------
# Then, we use the GM/WM boundary (GWB) and CSF/GM boundary (CGB) from
# CRUISE to compute cortical depth with a volume-preserving technique.
# Note that the number of layers is arbitrary, and does not reflect
# cytoarchitectonic layers. A good number in practice is 4, and up to
# for really high resolution (0.5mm and finer).
depth = nighres.laminar.volumetric_layering(
                        inner_levelset=cruise['gwb'],
                        outer_levelset=cruise['cgb'],
                        n_layers=4,
                        save_data=True,
                        output_dir=out_dir)

###########################################################################
# Now we look at the laminar depth estimates
if not skip_plots:
    plotting.plot_img(depth['depth'],cut_coords=[-75.0,90.0,-30.0],
                      vmin=0, vmax=1, cmap='autumn',  colorbar=True,
                      annotate=False,  draw_cross=False)

############################################################################



#############################################################################
# Cortical surface inflation
# --------------------------------
# For display purposes, we create a surface mesh from the average cortical
# CRUISE surface, which we then inflate
cortical_surface = nighres.surface.levelset_to_mesh(
                        levelset_image=cruise['avg'],
                        save_data=True,
                        output_dir=out_dir)

inflated_surface = nighres.surface.surface_inflation(
                        surface_mesh=cortical_surface['result'],
                        save_data=True,
                        output_dir=out_dir)

#############################################################################



#############################################################################
# Laminar data sampling and cortical surface mapping
# --------------------------------
# To sample data along the cortical depth, you need to combine the depth estimates
# and the underlying contrast of interest, here the T1 map.
laminar_profile = nighres.laminar.profile_sampling(
                        profile_surface_image=depth['boundaries'], 
                        intensity_image=t1_file,
                        save_data=True,
                        output_dir=out_dir)

# Now we display the result on the original and inflated surface.
# We will use the T1 values sampled on  the layer boundaries, which is a 4D image (with the 4th
# dimension the location along the cortical depth). As a result, the mesh will have multiple
# values associated, which may not always be well handled by surface viewers.
# For simplicity, we extract the values from the middle depth layer boundary.
sample2_file = laminar_profile['result'].replace('.nii','_bound2.nii')
if not os.path.isfile(sample2_file):
    print("Extract values from boundary 2 (middle of the cortical depth)")
    sample2 = nighres.io.load_volume(laminar_profile['result'])
    sample2 = nibabel.Nifti1Image(sample2.get_fdata()[:,:,:,2], sample2.affine, sample2.header)
    nighres.io.save_volume(sample2_file, sample2)
    
mapped_surface = nighres.surface.surface_mesh_mapping(
                        intensity_image=sample2_file, 
                        surface_mesh=cortical_surface['result'], 
                        inflated_mesh=inflated_surface['result'],
                        mapping_method="highest_value",
                        save_data=True, 
                        output_dir=out_dir)

#############################################################################

    
#############################################################################
# If the example is not run in a jupyter notebook, render the plots:
if not skip_plots:
    plotting.show()

#############################################################################
# Now we plot the surface results
if not skip_3d:
    surface = pyvista.read(mapped_surface['original'])
    surface.plot(interactive=True,window_size=[256,256])
    
    surface = pyvista.read(mapped_surface['inflated'])
    surface.plot(interactive=True,window_size=[256,256])
    
#############################################################################
# References
# -----------
# .. [1] Bogovic, J.A., Prince, J.L., Bazin, P.-L., 2013. A multiple object 
#        geometric deformable model for image segmentation. Computer Vision and Image 
#        Understanding 117, 145–157. https://doi.org/10.1016/j.cviu.2012.10.006
# .. [2] Han, X., Pham, D.L., Tosun, D., Rettmann, M.E., Xu, C., Prince, J.L., 2004. 
#        CRUISE: Cortical reconstruction using implicit surface evolution. 
#        NeuroImage 23, 997–1012. https://doi.org/10.1016/j.neuroimage.2004.06.043
# .. [3] Waehnert, M.D., Dinse, J., Weiss, M., Streicher, M.N., Waehnert, P., Geyer, S., 
#        Turner, R., Bazin, P.-L., 2014. Anatomically motivated modeling of cortical laminae. 
#        NeuroImage 93, 210–220. https://doi.org/10.1016/j.neuroimage.2013.03.078
# .. [4] Tosun, D., Rettmann, M., Prince, J., 2004. Mapping techniques for aligning sulci 
#        across multiple brains. Medical Image Analysis 8, 295–309. 
#        https://doi.org/10.1016/j.media.2004.06.020
