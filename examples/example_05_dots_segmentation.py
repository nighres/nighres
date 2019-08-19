"""
DOTS white matter segmentation
==============================

This example shows how to perform white matter segmentation using Diffusion
Oriented Tract Segmentation (DOTS) algorithm [1]_:

1. Downloading DOTS atlas prior informationg using
    :func:`nighres.data.download_DOTS_atlas`
2. Downloading an example DTI data set using
    :func:`nighres.data.download_DTI_2mm`
3. Segmenting white matter in DTI images into major tracts using 
    :func:`nighres.brain.dots_segmentation` [1]_
4. Visualizing the results using matplotlib

"""

############################################################################
# Import and download
# -------------------
# First we import ``nighres`` and ``os`` to set the output directory. Make sure 
# to run this file in a  directory you have write access to, or change the 
# ``out_dir`` variable below. We can downloadthe  DOTS atlas priors and an 
# example DTI dataset using the following command. The registration step of the 
# DOTS function relies on ``dipy``, so make sure you have installed it 
# (https://nipy.org/dipy/).

import os
import nighres

in_dir = os.path.join(os.getcwd(), 'nighres_examples/data_sets')
out_dir = os.path.join(os.getcwd(), 'nighres_examples/dots_segmentation')
nighres.data.download_DOTS_atlas()
dataset = nighres.data.download_DTI_2mm(in_dir)

############################################################################
# White matter segmentation
# -------------------------
# The DOTS segmentation can be run as follows. By default, the algorithm uses
# the tract atlas consisting of 23 tracts specified in [2]_. This can be changed
# to the full atlas by changing the value of the parameter 'wm_atlas' to 2. 
# Please see documentation for details.

dots_results = nighres.brain.dots_segmentation(tensorimg=dataset['dti'],
                                               mask=dataset['mask'],
                                               save_data=True,
                                               output_dir=out_dir,
                                               file_name='DOTS_results.nii.gz')
segmentation = dots_results['segmentation']
energy = dots_results['energy']

############################################################################
# .. tip:: the parameter s_I controls how isotropic label energies propagate 
#    to their neighborhood and can have a significant effect on tract volume.
#    Similarly, the value of the parameter convergence_threshold has an effect
#    on the results. Experiment with changin their values.

#############################################################################
# Interpretation of results
# -------------------------
# The integers in the segmentation array correspond to the tracts specified in
# atlas_labels_1 (in case of using wm_atlas 1) which can be imported as follows

from nighres.brain.dots_segmentation import atlas_labels_1

#############################################################################
# Visualization of results
# ------------------------
# We can visualize the segmented tracts overlaid on top of a fractional 
# anisotropy map. Let's first import the necessary modules and define a
# colormap. Then, we calculate the FA map and show the tracts. Let's also
# calculate a posterior probability and show an individual tract.

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

# This defines the following colormap
# transparent = isotropic
# semi-transparent red = other white matter
# opaque colours = individual tracts
# white = overlapping tracts

N_t = 23
N_o = 50
newcolors = np.zeros((N_t + N_o, 4))
newcolors[0,:] = np.array([.2, .2, .2, 0])
newcolors[1,:] = np.array([1, 0, 0, .25])
rainbow = cm.get_cmap('rainbow', N_t - 2)
newcolors[2:N_t,:] = rainbow(np.linspace(0, 1, N_t - 2))
newcolors[N_t::,:] = np.ones(newcolors[N_t::,:].shape)
newcmp = ListedColormap(newcolors)

# Calculate FA
tensor_volume = nb.load(in_dir + 'DTI_2mm.nii.gz').get_data()
xs, ys, zs, _ = tensor_volume.shape
tenfit = np.zeros((xs, ys, zs, 3, 3))
tenfit[:,:,:,0,0] = tensor_volume[:,:,:,0]
tenfit[:,:,:,1,1] = tensor_volume[:,:,:,1]
tenfit[:,:,:,2,2] = tensor_volume[:,:,:,2]
tenfit[:,:,:,0,1] = tensor_volume[:,:,:,3]
tenfit[:,:,:,1,0] = tensor_volume[:,:,:,3]
tenfit[:,:,:,0,2] = tensor_volume[:,:,:,4]
tenfit[:,:,:,2,0] = tensor_volume[:,:,:,4]
tenfit[:,:,:,1,2] = tensor_volume[:,:,:,5]
tenfit[:,:,:,2,1] = tensor_volume[:,:,:,5]
tenfit[np.isnan(tenfit)] = 0
evals, evecs = np.linalg.eig(tenfit)   
R = tenfit / np.trace(tenfit, axis1=3, axis2=4)[:,:,:,np.newaxis,np.newaxis]
FA = np.sqrt(0.5 * (3 - 1/(np.trace(np.matmul(R,R), axis1=3, axis2=4))))
FA[np.isnan(FA)] = 0

# Show segmentation
fig, ax = plt.subplots(1, 3, figsize=(28,5))
ax[0].imshow(np.rot90(FA[:,55,:]), cmap = 'gray', vmin = 0, vmax = 1)
ax[0].imshow(np.rot90(segmentation[:,55,:]), cmap=newcmp, alpha=.9)
ax[1].imshow(np.rot90(FA[:,:,30]), cmap = 'gray', vmin = 0, vmax = 1)
ax[1].imshow(np.rot90(segmentation[:,:,30]), cmap=newcmp, alpha=.9)
ax[2].imshow(np.rot90(FA[60,:,:]), cmap = 'gray', vmin = 0, vmax = 1)
ax[2].imshow(np.rot90(segmentation[60,:,:]), cmap=newcmp, alpha=.9)
for i in range(3):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
fig.tight_layout()
fig.savefig('CST_posterior11.png')
plt.show()

############################################################################
# .. image:: ../_static/dots_hard_segmentation.png
#############################################################################

#############################################################################
# Visualization of posterior probabilities
# ----------------------------------------
# We can visualize the posterior probability of a tract of interest in the 
# following way. First, let's import the array tract_pair_sets_1 (or 2 in case
# of using wm_atlas 2). Then, let's import and run the function 
# calc_posterior_probability

from nighres.brain.dots_segmentation import tract_pair_sets_1
from nighres.brain.dots_segmentation import calc_posterior_probability

# Select the corticospinal tract in the right hemisphere
tract_idx = 10

# Calculate posterior probability 
p_l = calc_posterior_probability(tract_idx, energy, 1, tract_pair_sets_1)

# Show results
p_l[p_l == 0] = np.nan
fig, ax = plt.subplots(1, 3, figsize=(28,5))
ax[0].imshow(np.rot90(FA[:,55,:]), cmap = 'gray', vmin = 0, vmax = 1)
ax[0].imshow(np.rot90(p_l[:,55,:]), alpha = .75)
ax[1].imshow(np.rot90(FA[:,:,30]), cmap = 'gray', vmin = 0, vmax = 1)
ax[1].imshow(np.rot90(p_l[:,:,30]), alpha=.75)
ax[2].imshow(np.rot90(FA[60,:,:]), cmap = 'gray', vmin = 0, vmax = 1)
ax[2].imshow(np.rot90(p_l[60,:,:]), alpha=.75)
for i in range(3):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
fig.tight_layout()
fig.savefig('CST_posterior.png')
plt.show()

############################################################################
# .. image:: ../_static/dots_posterior_probability.png
#############################################################################

#############################################################################
# References
# ----------
# .. [1] Bazin, Pierre-Louis, et al. "Direct segmentation of the major white 
#    matter tracts in diffusion tensor images." Neuroimage (2011)
#    doi: https://doi.org/10.1016/j.neuroimage.2011.06.020
# .. [2] Bazin, Pierre-Louis, et al. "Efficient MRF segmentation of DTI white 
#    matter tracts using an overlapping fiber model." Proceedings of the 
#    International Workshop on Diffusion Modelling and Fiber Cup (2009)
