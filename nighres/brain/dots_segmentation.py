import os
import sys
import time
import numpy as np
import nibabel as nb
from ..io import load_volume, save_volume
from dipy.core.gradients import gradient_table
from dipy.align.transforms import AffineTransform3D
from dipy.align.imaffine import MutualInformationMetric, AffineRegistration
from nighres.global_settings import ATLAS_DIR


atlas_labels_1 = ['isotropic', 'other_WM', 'ATR_L', 'ATR_R', 'CC_front', 
                  'CC_post', 'CC_sup', 'CG_L', 'CG_R', 'CST_L', 'CST_R', 
                  'IFO_L', 'IFO_R', 'ILF_L', 'ILF_R', 'ML_L', 'ML_R', 'OPR_L', 
                  'OPR_R', 'SLF_L', 'SLF_R', 'UNC_L', 'UNC_R']

tract_pair_sets_1 = [{2, 4}, {3, 4}, {2, 6}, {3, 6}, {4, 6}, {4, 7}, {5, 7}, 
                     {6, 7}, {8, 4}, {8, 5}, {8, 6}, {9, 6}, {10, 6}, {9, 10}, 
                     {2, 11}, {11, 4}, {11, 5}, {3, 12}, {12, 4}, {12, 5}, 
                     {10, 12}, {13, 5}, {11, 13}, {5, 14}, {12, 14}, {2, 15}, 
                     {9, 15}, {10, 15}, {16, 9}, {16, 10}, {16, 15}, {17, 5}, 
                     {17, 11}, {17, 13}, {18, 5}, {18, 12}, {18, 14}, {19, 11}, 
                     {19, 13}, {17, 19}, {10, 20}, {20, 12}, {20, 14}, {18, 20}, 
                     {2, 21}, {4, 21}, {11, 21}, {21, 13}, {12, 22}, {22, 14}]

atlas_labels_2 = ['isotropic', 'other_WM', 'ATR_L', 'ATR_R', 'CC_front', 
                  'CC_post', 'CC_sup', 'CG_L', 'CG_R', 'CPT_L', 'CPT_R', 
                  'CST_L', 'CST_R', 'FNX_L', 'FNX_R', 'ICP_L', 'ICP_R', 'IFO_L', 
                  'IFO_R', 'ILF_L', 'ILF_R', 'MCP', 'ML_L', 'ML_R', 'OPR_L', 
                  'OPR_R', 'OPT_L', 'OPT_R', 'PTR_L', 'PTR_R', 'SCP_L', 'SCP_R', 
                  'SFO_L', 'SFO_R', 'SLF_L', 'SLF_R', 'STR_L', 'STR_R', 'TAP', 
                  'UNC_L', 'UNC_R']

tract_pair_sets_2 = [{2, 4}, {3, 4}, {2, 6}, {3, 6}, {4, 6}, {4, 7}, {5, 7}, 
                     {6, 7}, {8, 4}, {8, 5}, {8, 6}, {9, 2}, {9, 6}, {10, 3}, 
                     {10, 6}, {9, 10}, {11, 6}, {9, 11}, {10, 11}, {12, 6}, 
                     {9, 12}, {10, 12}, {11, 12}, {2, 13}, {13, 7}, {9, 13}, 
                     {5, 14}, {8, 14}, {11, 15}, {16, 12}, {17, 2}, {17, 4}, 
                     {17, 5}, {17, 9}, {17, 13}, {18, 3}, {18, 4}, {18, 5}, 
                     {18, 10}, {18, 12}, {18, 14}, {19, 5}, {9, 19}, {19, 13}, 
                     {17, 19}, {20, 5}, {20, 14}, {18, 20}, {9, 21}, {10, 21}, 
                     {11, 21}, {12, 21}, {21, 15}, {16, 21}, {2, 22}, {9, 22}, 
                     {11, 22}, {12, 22}, {22, 15}, {16, 22}, {21, 22}, {10, 23}, 
                     {11, 23}, {12, 23}, {16, 23}, {21, 23}, {22, 23}, {24, 5}, 
                     {24, 9}, {24, 13}, {24, 17}, {24, 19}, {25, 5}, {25, 10}, 
                     {25, 14}, {25, 18}, {25, 20}, {9, 26}, {26, 11}, {26, 13}, 
                     {17, 26}, {10, 27}, {27, 12}, {27, 14}, {18, 27}, {27, 20}, 
                     {25, 27}, {28, 5}, {9, 28}, {28, 13}, {17, 28}, {19, 28}, 
                     {24, 28}, {29, 5}, {29, 6}, {8, 29}, {10, 29}, {12, 29},
                     {29, 14}, {18, 29}, {20, 29}, {25, 29}, {27, 29}, {2, 30}, 
                     {9, 30}, {11, 30}, {30, 15}, {21, 30}, {30, 22}, {30, 23}, 
                     {16, 31}, {21, 31}, {22, 31}, {31, 23}, {32, 2}, {32, 4}, 
                     {32, 6}, {32, 9}, {32, 11}, {32, 13}, {32, 17}, {32, 28}, 
                     {33, 3}, {33, 4}, {33, 6}, {33, 10}, {33, 14}, {33, 18}, 
                     {33, 29}, {9, 34}, {17, 34}, {34, 19}, {24, 34}, {34, 28}, 
                     {10, 35}, {35, 12}, {18, 35}, {35, 20}, {25, 35}, {35, 29}, 
                     {2, 36}, {36, 6}, {9, 36}, {11, 36}, {17, 36}, {26, 36}, 
                     {32, 36}, {34, 36}, {3, 37}, {37, 6}, {10, 37}, {12, 37}, 
                     {18, 37}, {27, 37}, {37, 29}, {33, 37}, {35, 37}, {5, 38}, 
                     {38, 7}, {8, 38}, {9, 38}, {10, 38}, {13, 38}, {38, 14}, 
                     {17, 38}, {18, 38}, {19, 38}, {20, 38}, {24, 38}, {25, 38}, 
                     {27, 38}, {28, 38}, {29, 38}, {34, 38}, {35, 38}, {2, 39}, 
                     {4, 39}, {13, 39}, {17, 39}, {19, 39}, {38, 39}, {40, 14}, 
                     {40, 18}, {40, 20}, {40, 27}]

def _theta(v_1, v_2):
    # Return angle between vectors v_1 and v_2 normalized to be in [0,1]
    angle = (2/np.pi) * np.arccos(np.abs(np.dot(v_1, v_2)))
    return angle


def _calc_s_T(i, j, k, a, b, c, evecs, v_xy):
    # Return connectivity between voxels i,j,k and a,b,c assuming a single 
    # tract (Eq 5)
    s_T = (1 - np.nanmin([_theta(evecs[i,j,k,:,0], v_xy[a-i+1,b-j+1,c-k+1,:]),
           _theta(evecs[a,b,c,:,0], v_xy[a-i+1,b-j+1,c-k+1,:])])) * \
          (1 - 2*_theta(evecs[i,j,k,:,0], evecs[a,b,c,:,0]))
    return s_T

    
def _calc_s_O(i, j, k, a, b, c, evals, evecs, v_xy):
    # Return connectivity between voxels i,j,k and a,b,c assuming
    # overlapping tracts (Eq 6)
    temp = np.zeros((4,2,3))
    temp[0,0,:] = evecs[i,j,k,:,0]
    temp[0,1,:] = evecs[a,b,c,:,0]
    temp[1,0,:] = evals[i,j,k,1] / evals[i,j,k,0] * evecs[i,j,k,:,1]
    temp[1,1,:] = evecs[a,b,c,:,0]
    temp[2,0,:] = evecs[i,j,k,:,0]
    temp[2,1,:] = evals[a,b,c,1] / evals[a,b,c,0] * evecs[a,b,c,:,1]
    temp[3,0,:] = evals[i,j,k,1] / evals[i,j,k,0] * evecs[i,j,k,:,1]
    temp[3,1,:] = evals[a,b,c,1] / evals[a,b,c,0] * evecs[a,b,c,:,1]
    vals = np.zeros(4)
    for x in range(4):
        vals[x] = _theta(temp[x,0,:], temp[x,1,:])
    v_O = temp[np.argmin(vals),:,:]
    s_O = (1 - np.nanmin([_theta(v_O[0,:], v_xy[a-i+1,b-j+1,c-k+1,:]),
          _theta(v_O[1,:], v_xy[a-i+1,b-j+1,c-k+1,:])])) * \
          (1 - 2*_theta(v_O[0,:], v_O[1,:]))
    return s_O


def _half_pos_nhood(i, j, k, evecs, v_xy):
    # Return half-neighborhood where dot product between principal diffusion
    # direction of voxel i,j,k and the direction between neighboring voxels
    # is positive
    pos_indices = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if np.dot(evecs[i,j,k,:,0], v_xy[a,b,c]) > 0:
                    pos_indices.append(np.array([a,b,c]))
    return np.array(pos_indices) + [i,j,k] - [1,1,1]


def _half_neg_nhood(i, j, k, evecs, v_xy):
    # Return half-neighborhood where dot product between principal diffusion
    # direction of voxel i,j,k and the direction between neighboring voxels
    # is negative
    neg_indices = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if np.dot(evecs[i,j,k,:,0], v_xy[a,b,c]) < 0:
                    neg_indices.append(np.array([a,b,c]))         
    return np.array(neg_indices) + [i,j,k] - [1,1,1]


def _calc_x_plus(i, j, k, evecs, v_xy):
    # Return x^+
    pos_indices = _half_pos_nhood(i,j,k,evecs,v_xy)
    max_s_T = -np.inf
    for idx in range(pos_indices.shape[0]):
        temp_s_T = _calc_s_T(i,j,k,pos_indices[idx,0],
                            pos_indices[idx,1],pos_indices[idx,2],
                            evecs, v_xy)
        if temp_s_T > max_s_T:
            max_s_T = temp_s_T
            argmax = pos_indices[idx,:]
    return argmax


def _calc_x_minus(i, j, k, evecs, v_xy):
    # Return x^-
    neg_indices = _half_neg_nhood(i,j,k,evecs,v_xy)
    max_s_T = -np.inf
    for idx in range(neg_indices.shape[0]):
        temp_s_T = _calc_s_T(i,j,k,neg_indices[idx,0],
                            neg_indices[idx,1],neg_indices[idx,2],
                            evecs, v_xy)
        if temp_s_T > max_s_T:
            max_s_T = temp_s_T
            argmax = neg_indices[idx,:]
    return argmax


def _calc_c_l(i, j, k, l, m, evecs, fiber_dir):
    # Return direction index (Eqs 8, 9)
    if m == None:
        c_l = np.linalg.norm(fiber_dir[i,j,k,:,l]) * (1 - 2 *
              _theta(evecs[i,j,k,:,0], fiber_dir[i,j,k,:,l] /
              np.linalg.norm(fiber_dir[i,j,k,:,l])))
    else:
        comp_dir = np.stack((fiber_dir[i,j,k,:,l] + fiber_dir[i,j,k,:,m],
                             fiber_dir[i,j,k,:,l] - fiber_dir[i,j,k,:,m]))
        comp_dir = comp_dir[np.argmax(np.linalg.norm(comp_dir, axis=1)),:]
        comp_dir = (comp_dir / np.linalg.norm(comp_dir) * 
                    (np.linalg.norm(fiber_dir[i,j,k,:,l]) +
                     np.linalg.norm(fiber_dir[i,j,k,:,m])) / 2)
        c_l = np.linalg.norm(comp_dir) * (1 - 2 * 
              _theta(evecs[i,j,k,:,0], comp_dir /
              np.linalg.norm(comp_dir)))
    return c_l


def _calc_V1(d_T, d_O, d_I, u_l, u_lm, c_l, c_lm, c_I, fiber_p, 
             tract_pair_sets, N_t, N_o, brain_mask):
    # Return energy using the unary term only (Eq 11)
    xs, ys, zs = d_T.shape
    MRF_V1 = np.zeros((xs, ys, zs, N_t + N_o))
    print('Calculating V1')
    for i in range(xs):
        print(str(np.round((i / xs)*100, 0)) + ' %', end="\r")
        for j in range(ys):
            for k in range(zs):
                if brain_mask[i,j,k]:
                        
                    # Calculate isotropic energy
                    MRF_V1[i,j,k,0] = c_I * d_I[i,j,k] * u_l[i,j,k,0]
                        
                    # Calculate individual tract energies
                    for idx in range(1,N_t):
                        l = idx
                        if fiber_p[i,j,k,l] == 0:
                            MRF_V1[i,j,k,idx] = np.nan
                        else:
                            MRF_V1[i,j,k,idx] = d_T[i,j,k] * u_l[i,j,k,l] * \
                                                c_l[i,j,k,l]

                    # Calculate overlapping tract energies
                    for idx in range(N_t,N_t+N_o):
                        l,m = tract_pair_sets[idx - N_t]
                        if fiber_p[i,j,k,l] == 0 or fiber_p[i,j,k,m] == 0:
                            MRF_V1[i,j,k,idx] = np.nan
                        else:
                            MRF_V1[i,j,k,idx] = d_O[i,j,k] * \
                                                u_lm[i,j,k,idx-N_t] * \
                                                c_lm[i,j,k,idx-N_t]
    return MRF_V1
            

def _calc_U(prev_iter_U, d_T, d_O, d_I, u_l, u_lm, c_l, c_lm, c_I, fiber_p, 
            tract_pair_sets, s_I, s_T_x_p, s_T_x_m, s_O_x_m, s_O_x_p, x_p, x_m, 
            brain_mask, N_t, N_o):
    # Return total energy (Eq 12)
    tract_pair_array = np.array([list(i) for i in tract_pair_sets])
    xs, ys, zs = d_T.shape
    curr_U = np.zeros((xs, ys, zs, N_t + N_o))
    
    for i in range(xs):
        print(str(np.round((i / xs)*100, 0)) +   ' %', end="\r")
        for j in range(ys):
            for k in range(zs):
                if brain_mask[i,j,k]:
                    
                    # Check isotropic energy
                    curr_U[i,j,k,0] = c_I * d_I[i,j,k] * u_l[i,j,k,0] + \
                                      (s_I * (np.nansum(prev_iter_U[i-1:i+2,
                                       j-1:j+2,k-1:k+2,0]) - prev_iter_U[i,j,
                                       k,0]) / 26)
                    
                    
                    # Check individual tract energy
                    for l in range(1,N_t):
                        if fiber_p[i,j,k,l] == 0:
                            curr_U[i,j,k,l] = np.nan
                        else:
                            idx = np.concatenate(([l], np.where(np.any(
                                                 tract_pair_array
                                                 == l, axis=1))[0]+N_t))
                            curr_U[i,j,k,l] = d_T[i,j,k] * u_l[i,j,k,l] * \
                                              c_l[i,j,k,l] + \
                                              0.5 * (s_T_x_p[i,j,k] * 
                                              np.nanmax(prev_iter_U[
                                                        x_p[i,j,k,0],
                                                        x_p[i,j,k,1],
                                                        x_p[i,j,k,2],
                                                        idx])) + \
                                              0.5 * (s_T_x_m[i,j,k] * 
                                              np.nanmax(prev_iter_U[
                                                        x_m[i,j,k,0],
                                                        x_m[i,j,k,1],
                                                        x_m[i,j,k,2],
                                                        idx]))
                                                
                    # Check overlapping tract energy
                    for idx in range(N_t,N_t+N_o):
                        l,m = tract_pair_sets[idx-N_t]
                        if fiber_p[i,j,k,l] == 0 or fiber_p[i,j,k,m] == 0:
                            curr_U[i,j,k,idx] = np.nan
                        else:
                            curr_U[i,j,k,idx] = d_O[i,j,k] * \
                                                u_lm[i,j,k,idx-N_t] * \
                                                c_lm[i,j,k,idx-N_t] + \
                                                0.5 * (s_O_x_p[i,j,k] *
                                                np.nanmax([prev_iter_U[
                                                           x_p[i,j,k,0],
                                                           x_p[i,j,k,1],
                                                           x_p[i,j,k,2],
                                                           idx],
                                                           prev_iter_U[
                                                           x_p[i,j,k,0],
                                                           x_p[i,j,k,1],
                                                           x_p[i,j,k,2],
                                                           l],
                                                           prev_iter_U[
                                                           x_p[i,j,k,0],
                                                           x_p[i,j,k,1],
                                                           x_p[i,j,k,2],
                                                           m]])) + \
                                               0.5 * (s_O_x_m[i,j,k] *
                                               np.nanmax([prev_iter_U[
                                                          x_m[i,j,k,0],
                                                          x_m[i,j,k,1],
                                                          x_m[i,j,k,2],
                                                          idx],
                                                          prev_iter_U[
                                                          x_m[i,j,k,0],
                                                          x_m[i,j,k,1],
                                                          x_m[i,j,k,2],
                                                          l],
                                                          prev_iter_U[
                                                          x_m[i,j,k,0],
                                                          x_m[i,j,k,1],
                                                          x_m[i,j,k,2],
                                                          m]]))
    return curr_U


def _calc_segmentation(U):
    # Return hard segmentation based on MRF energy U  
    U_temp = np.copy(U)
    U_temp[np.isnan(U_temp)] = -np.inf
    segmentation = np.argmax(U_temp, axis = 3)
    return segmentation


def calc_posterior_probability(l, U, wm_atlas, tract_pair_sets, g0 = None):
    # Return posterior probability of tract l from MRF energy U (Eq 15)               
    if wm_atlas == 1:
        N_t = 23
    elif wm_atlas == 2:
        N_t = 41
    if g0 == None:
        g0 = 1 / N_t
    tract_pair_array = np.array([list(i) for i in tract_pair_sets])
    idx = np.concatenate(([l], np.where(np.any(tract_pair_array 
                                               == l, axis=1))[0]+N_t))
    posterior_l = (np.nansum(np.exp(g0*U[:,:,:,idx]), axis=3) /
                   np.nansum(np.exp(g0*U),axis=3))
    return posterior_l
    

def dots_segmentation(tensorimg, mask, wm_atlas = 1, max_iter = 25,
                      convergence_threshold = 0.002, g0 = 1/42, s_I = 1/42,
                      save_data = False, overwrite = False, output_dir = None, 
                      file_name = None):
    """DOTS segmentation

    Segment major white matter tracts in diffusion tensor images using Diffusion
    Oriented Tract Segmentation (DOTS) algorithm.
    
    Parameters
    ----------
    tensorimg: niimg
        Input image containing the diffusion tensor coefficients in the
        following order: volumes 0-5: D11, D22, D33, D12, D13, D23
    mask: niimg
        Binary brain mask image which limits computation to the defined volume.
    wm_atlas: int, optional
        Define which white matter atlas to use. Option 1 for 23 tracts [2]_ 
        and option 2 for 39 tracts [1]_. (default is 1)
    max_iter: int, optional
        Maximum number of iterations in the conditional modes algorithm.
        (default is 20)
    convergence_threshold: float, optional
        Threshold for when the iterated conditonal modes algorithm is considered
        to have converged. Defined as the fraction of labels that change during
        one step of the algorithm. (default is 0.002)
    g0: float, optional
        Parameter controlling the sharpness of the membership function. (default
        is 1/42)
    s_I: float, optional
        Parameter controlling how isotropic label energies propagate to their
        neighborhood. (default is 1/42)
    save_data: bool, optional
        Save output data to file. (default is False)
    overwrite: bool, optional
        Overwrite existing results. (default is False)
    output_dir: str, optional
        Path to desired output directory, will be created if it doesn't exist.
    file_name: str, optional
        Desired base name for output files with file extension, suffixes will 
        be added.

    Returns
    ----------
    dict
        Dictionary collecting outputs under the following keys
        (suffix of output files in brackets)

        * segmentation (niimg): Hard segmentation of white matter.
        * energy (niimg): Markov Random Field energies of labels.
        
    Notes
    ----------
    Algorithm details can be found in [1]_.

    References
    ----------
    .. [1] Bazin, Pierre-Louis, et al. "Direct segmentation of the major white 
       matter tracts in diffusion tensor images." Neuroimage (2011)
       doi: https://doi.org/10.1016/j.neuroimage.2011.06.020
    .. [2] Bazin, Pierre-Louis, et al. "Efficient MRF segmentation of DTI white 
       matter tracts using an overlapping fiber model." Proceedings of the 
       International Workshop on Diffusion Modelling and Fiber Cup (2009)
    """
    
    # Ignore runtime warnings that arise from trying to divide by 0/nan
    # and all nan slices
    np.seterr(divide = 'ignore', invalid = 'ignore')
    
    # Define the scalar constant c_I
    c_I = 1/2    
    
      
    # Create an array containing the directions between neighbors 
    v_xy = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if (i,j,k) == (1,1,1):
                    v_xy[i,j,k,:] = np.nan
                else:
                    x = np.array([1,0,0])
                    y = np.array([0,1,0])
                    z = np.array([0,0,1])
                    c = np.array([1,1,1])
                    v_xy[i,j,k,:] = i*x + y*j + z*k - c
                    v_xy[i,j,k,:] = v_xy[i,j,k,:] / \
                                    np.linalg.norm(v_xy[i,j,k,:])
    
    
    # Load tensor image
    tensor_volume = load_volume(tensorimg).get_data()
    
    
    # Load brain mask
    brain_mask = load_volume(mask).get_data()
    
    
    # Get dimensions of diffusion data
    xs, ys, zs, _ = tensor_volume.shape
    DWI_affine = load_volume(tensorimg).affine
    
    
    # Calculate diffusion tensor eigenvalues and eigenvectors
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
    idx = np.flip(np.argsort(evals, axis = 3), axis = 3)
    evals = np.flip(np.sort(evals, axis = 3), axis = 3)
    for i in range(xs):
        for j in range(ys):
            for k in range(zs):
                evecs[i,j,k,:,:] = evecs[i,j,k,:,idx[i,j,k,:]]
    evals[~brain_mask] = 0
    evecs[~brain_mask] = 0

    # Calculate FA
    R = tenfit / np.trace(tenfit, axis1=3, axis2=4)[:,:,:,np.newaxis,np.newaxis]
    FA = np.sqrt(0.5 * (3 - 1/(np.trace(np.matmul(R,R), axis1=3, axis2=4))))
    FA[np.isnan(FA)] = 0    

    
    if wm_atlas == 1:
                    
        # Use smaller atlas
        # Indices are
        # 0 for isotropic regions
        # 1 for unclassified white matter 
        # 2-22 for individual tracts
        # 22-73 for overlapping tracts
        N_t = 23
        N_o = 50        
        print('N_t, N_o = '+str(N_t)+', '+str(N_o))
        atlas_path = os.path.join(ATLAS_DIR, 'DOTS_atlas')
        fiber_p = nb.load(os.path.join(atlas_path,'fiber_p.nii.gz')).get_data()
        max_p = np.nanmax(fiber_p[:,:,:,2::], axis = 3)
        fiber_dir = nb.load(os.path.join(atlas_path, 'fiber_dir.nii.gz')
                            ).get_data()
        atlas_affine = nb.load(os.path.join(atlas_path,'fiber_p.nii.gz')).affine
        del_idx = [9,10,13,14,15,16,21,26,27,28,29,30,31,32,33,36,37,38]
        fiber_p = np.delete(fiber_p, del_idx, axis = 3)
        fiber_dir = np.delete(fiber_dir, del_idx, axis = 4)
        tract_pair_sets = [{2, 4}, {3, 4}, {2, 6}, {3, 6}, {4, 6}, {4, 7}, 
                           {5, 7}, {6, 7}, {8, 4}, {8, 5}, {8, 6}, {9, 6}, 
                           {10, 6}, {9, 10}, {2, 11}, {11, 4}, {11, 5}, {3, 12}, 
                           {12, 4}, {12, 5}, {10, 12}, {13, 5}, {11, 13},
                           {5, 14}, {12, 14}, {2, 15}, {9, 15}, {10, 15}, 
                           {16, 9}, {16, 10}, {16, 15}, {17, 5}, {17, 11}, 
                           {17, 13}, {18, 5}, {18, 12}, {18, 14}, {19, 11}, 
                           {19, 13}, {17, 19}, {10, 20}, {20, 12}, {20, 14},
                           {18, 20}, {2, 21}, {4, 21}, {11, 21}, {21, 13}, 
                           {12, 22}, {22, 14}]
        tract_pair_array = np.array([list(i) for i in tract_pair_sets])
    
    elif wm_atlas == 2:
    
        # Use full atlas
        # Indices are
        # 0 for isotropic regions
        # 1 for unclassified white matter 
        # 2-40 for individual tracts
        # 41-224 for overlapping tracts
        N_t = 41
        N_o = 185
        atlas_path = os.path.join(ATLAS_DIR, 'DOTS_atlas')
        fiber_p = nb.load(os.path.join(atlas_path,'fiber_p.nii.gz')).get_data()
        max_p = np.nanmax(fiber_p[:,:,:,2::], axis = 3)
        fiber_dir = nb.load(os.path.join(atlas_path, 'fiber_dir.nii.gz')
                            ).get_data()
        atlas_affine = nb.load(os.path.join(atlas_path,'fiber_p.nii.gz')).affine
        tract_pair_sets = [{2, 4}, {3, 4}, {2, 6}, {3, 6}, {4, 6}, {4, 7}, 
                           {5, 7}, {6, 7}, {8, 4}, {8, 5}, {8, 6}, {9, 2}, 
                           {9, 6}, {10, 3}, {10, 6}, {9, 10}, {11, 6}, {9, 11}, 
                           {10, 11}, {12, 6}, {9, 12}, {10, 12}, {11, 12},
                           {2, 13}, {13, 7}, {9, 13}, {5, 14}, {8, 14}, 
                           {11, 15}, {16, 12}, {17, 2}, {17, 4}, {17, 5}, 
                           {17, 9}, {17, 13}, {18, 3}, {18, 4}, {18, 5}, 
                           {18, 10}, {18, 12}, {18, 14}, {19, 5}, {9, 19}, 
                           {19, 13}, {17, 19}, {20, 5}, {20, 14}, {18, 20}, 
                           {9, 21}, {10, 21}, {11, 21}, {12, 21}, {21, 15}, 
                           {16, 21}, {2, 22}, {9, 22}, {11, 22}, {12, 22},
                           {22, 15}, {16, 22}, {21, 22}, {10, 23}, {11, 23}, 
                           {12, 23}, {16, 23}, {21, 23}, {22, 23}, {24, 5}, 
                           {24, 9}, {24, 13}, {24, 17}, {24, 19}, {25, 5}, 
                           {25, 10}, {25, 14}, {25, 18}, {25, 20}, {9, 26},
                           {26, 11}, {26, 13}, {17, 26}, {10, 27}, {27, 12}, 
                           {27, 14}, {18, 27}, {27, 20}, {25, 27}, {28, 5}, 
                           {9, 28}, {28, 13}, {17, 28}, {19, 28}, {24, 28}, 
                           {29, 5}, {29, 6}, {8, 29}, {10, 29}, {12, 29},
                           {29, 14}, {18, 29}, {20, 29}, {25, 29}, {27, 29}, 
                           {2, 30}, {9, 30}, {11, 30}, {30, 15}, {21, 30}, 
                           {30, 22}, {30, 23}, {16, 31}, {21, 31}, {22, 31}, 
                           {31, 23}, {32, 2}, {32, 4}, {32, 6}, {32, 9},
                           {32, 11}, {32, 13}, {32, 17}, {32, 28}, {33, 3}, 
                           {33, 4}, {33, 6}, {33, 10}, {33, 14}, {33, 18}, 
                           {33, 29}, {9, 34}, {17, 34}, {34, 19}, {24, 34}, 
                           {34, 28}, {10, 35}, {35, 12}, {18, 35}, {35, 20},
                           {25, 35}, {35, 29}, {2, 36}, {36, 6}, {9, 36}, 
                           {11, 36}, {17, 36}, {26, 36}, {32, 36}, {34, 36}, 
                           {3, 37}, {37, 6}, {10, 37}, {12, 37}, {18, 37}, 
                           {27, 37}, {37, 29}, {33, 37}, {35, 37}, {5, 38}, 
                           {38, 7}, {8, 38}, {9, 38}, {10, 38}, {13, 38}, 
                           {38, 14}, {17, 38}, {18, 38}, {19, 38}, {20, 38}, 
                           {24, 38}, {25, 38}, {27, 38}, {28, 38}, {29, 38}, 
                           {34, 38}, {35, 38}, {2, 39}, {4, 39}, {13, 39}, 
                           {17, 39}, {19, 39}, {38, 39}, {40, 14}, {40, 18}, 
                           {40, 20}, {40, 27}]
        tract_pair_array = np.array([list(i) for i in tract_pair_sets])

    print('Diffusion and atlas data loaded ')
    
    
    # Register atlas priors to DWI data with DiPy
    print('Registering atlas priors to DWI data')
    metric = MutualInformationMetric(nbins = 32, sampling_proportion = None)
    affreg = AffineRegistration(metric = metric,
                                level_iters = [10000,1000,100],
                                sigmas = [3.0,1.0,0.0],
                                factors = [4,2,1])
    transformation = affreg.optimize(FA, 
                                     max_p,
                                     AffineTransform3D(), 
                                     params0=None,
                                     static_grid2world=DWI_affine,
                                     moving_grid2world=atlas_affine,
                                     starting_affine='mass')
    reg_fiber_p = np.zeros((xs, ys, zs, fiber_p.shape[-1]))
    for i in range(fiber_p.shape[-1]):
        reg_fiber_p[:,:,:,i] = transformation.transform(fiber_p[:,:,:,i])
    fiber_p = reg_fiber_p
    reg_fiber_dir = np.zeros((xs, ys, zs, 3, fiber_dir.shape[-1]))
    for i in range(fiber_dir.shape[-1]):
        for j in range(3):
            reg_fiber_dir[:,:,:,j,i] = transformation.transform(
                                       fiber_dir[:,:,:,j,i])
    fiber_dir = reg_fiber_dir
    fiber_p[~brain_mask,0] = 1
    fiber_p[~brain_mask,1:] = np.nan
    fiber_dir[~brain_mask] = np.nan
    print('Finished registration of atlas priors to DWI data')


    # Calculate diffusion type indices
    print('Calculating d_T, d_O, d_I')
    d_T = (evals[:,:,:,0] - evals[:,:,:,1]) / evals[:,:,:,0]
    d_O = (evals[:,:,:,0] - evals[:,:,:,2]) / evals[:,:,:,0]
    d_I = evals[:,:,:,2] / evals[:,:,:,0]
    d_T[~brain_mask] = np.nan
    d_O[~brain_mask] = np.nan
    d_I[~brain_mask] = 1
    print('Finished calculating d_T, d_O, d_I')

        
    # Calculate xplus and xminus
    x_m = np.zeros((xs,ys,zs,3))
    x_p = np.zeros((xs,ys,zs,3))
    print('Calculating xp, xm')
    for i in range(1,xs-1):
        print(str(np.round((i / xs)*100, 0)) + ' %', end="\r")
        for j in range(1,ys-1):
            for k in range(1,zs-1):
                if brain_mask[i,j,k]:
                    x_m[i,j,k,:] = _calc_x_minus(i,j,k,evecs,v_xy)
                    x_p[i,j,k,:] = _calc_x_plus(i,j,k,evecs,v_xy)   
    x_p = x_p.astype(int)
    x_m = x_m.astype(int)
    print('Finished calculating xp, xm')
        
        
    # Calculate required s_I, s_T, and s_O
    s_T_x_m = np.zeros((xs,ys,zs))*np.nan
    s_T_x_p = np.zeros((xs,ys,zs))*np.nan
    s_O_x_m = np.zeros((xs,ys,zs))*np.nan
    s_O_x_p = np.zeros((xs,ys,zs))*np.nan
    print('Calculating s_T, s_O for xp, xm')
    for i in range(1,xs-1):
        print(str(np.round((i / xs)*100, 0)) + ' %', end="\r")
        for j in range(1,ys-1):
            for k in range(1,zs-1):
                if brain_mask[i,j,k]:
                    s_T_x_p[i,j,k] = _calc_s_T(i,j,k,x_p[i,j,k,0],
                                              x_p[i,j,k,1],x_p[i,j,k,2],
                                              evecs, v_xy)
                    s_T_x_m[i,j,k] = _calc_s_T(i,j,k,x_m[i,j,k,0],
                                              x_m[i,j,k,1],x_m[i,j,k,2],
                                              evecs, v_xy)
                    s_O_x_p[i,j,k] = _calc_s_O(i,j,k,x_p[i,j,k,0],
                                              x_p[i,j,k,1],x_p[i,j,k,2],
                                              evals, evecs, v_xy)
                    s_O_x_m[i,j,k] = _calc_s_O(i,j,k,x_m[i,j,k,0],
                                              x_m[i,j,k,1],x_m[i,j,k,2],
                                              evals, evecs, v_xy)
    print('Finished calculating s_T, s_O for xp, xm')
        
        
    # Calculate shape prior arrays
    print('Calculating u_l, u_lm')
    u_l = fiber_p**2 / np.nansum(fiber_p, axis=3)[:,:,:,np.newaxis]
    u_lm = np.zeros((xs, ys, zs, len(tract_pair_sets)))
    for idx in range(len(tract_pair_sets)):
        l,m = tract_pair_sets[idx]
        u_lm[:,:,:,idx] = fiber_p[:,:,:,l]*fiber_p[:,:,:,m]*(fiber_p[:,:,:,l]
                          + fiber_p[:,:,:,m]) / \
                          np.nansum(fiber_p, axis=3) 
    u_l[~brain_mask] = np.nan
    u_l[~brain_mask,0] = 1
    u_lm[~brain_mask] = np.nan
    # Only ROIs where p != 0 are of interest
    u_l[u_l == 0] = np.nan
    u_lm[u_lm == 0] = np.nan
    print('Finished calculating u_l, u_lm')
        
        
    # Calculate direction coefficients
    c_l = np.zeros((xs, ys, zs, N_t))*np.nan
    c_lm = np.zeros((xs, ys, zs, len(tract_pair_sets)))*np.nan
    print('Calculating c_l, c_lm')
    for i in range(xs):
        print(str(np.round((i / xs)*100, 0)) + ' %', end="\r")
        for j in range(ys):
            for k in range(zs):
                for l in range(1,N_t):
                    if fiber_p[i,j,k,l] != 0:
                        c_l[i,j,k,l] = _calc_c_l(i,j,k,l,None,evecs,fiber_dir)
                for idx in range(len(tract_pair_sets)):
                    l,m = tract_pair_sets[idx]
                    if fiber_p[i,j,k,l] != 0 and fiber_p[i,j,k,m] != 0:
                        c_lm[i,j,k,idx] = _calc_c_l(i,j,k,l,m,evecs,fiber_dir)
    print('Finished calculating c_l, c_lm')
                        

    # Calculate energy based on unary term only
    MRF_V1 = _calc_V1(d_T, d_O, d_I, u_l, u_lm, c_l, c_lm, c_I, fiber_p, 
                      tract_pair_sets, N_t, N_o, brain_mask)


    # Maximize U
    print('Maximizing U')
    curr_U = np.copy(MRF_V1)
    iteration = 0
    change_in_labels = np.inf
    while iteration < max_iter and change_in_labels > convergence_threshold:
        at = time.time()
        prev_U = np.copy(curr_U)
        prev_segmentation = _calc_segmentation(prev_U)
        iteration += 1
        print('Iteration '+str(iteration))
        
        curr_U = _calc_U(prev_U, d_T, d_O, d_I, u_l, u_lm, c_l, c_lm, c_I, 
                         fiber_p, tract_pair_sets, s_I, s_T_x_p, s_T_x_m, 
                         s_O_x_m, s_O_x_p, x_p, x_m, brain_mask, N_t, N_o)
        
        curr_segmentation = _calc_segmentation(curr_U)           
        change_in_labels = (np.nansum(prev_segmentation != curr_segmentation) /
                            np.nansum(brain_mask))
        bt = time.time()
        print('Iteration '+str(iteration)+' took '+str(bt-at)+' seconds')
        print('Total U = '+str(np.nansum(curr_U)))
        print('Fraction of changed labels = '+str(change_in_labels))
    print('Finished maximizing U') 


    # Save results
    if save_data:
        save_volume(file_name, nb.Nifti1Image(curr_segmentation, DWI_affine))


    # Return results
    return {'segmentation': curr_segmentation, 'energy': curr_U}


