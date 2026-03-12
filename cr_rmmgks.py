
import matplotlib.pyplot as plt
import numpy as np
import astra
from trips.utilities.phantoms import *
from venv import create
import pylops
from trips.solvers.MMGKS import *
from trips.utilities.helpers import *
from trips.utilities.operators import *

from typing import Optional, Dict, Any, List
from skimage.transform import resize
from datetime import datetime


from scipy.optimize import newton, minimize
import scipy.linalg as la
import scipy.optimize as op
from pylops import Identity, LinearOperator

import pickle
from typing import List, Tuple, Optional
from copy import deepcopy
from rmmgks import *

def compute_weights(Lx, D, d_prev, q, d_limit=1e-3):

    weighted_grad_flat = (D @ Lx).flatten()
    
    # Normalization
    norm_inf = np.max(np.abs(weighted_grad_flat))
    g = np.abs(weighted_grad_flat) / norm_inf
    
    # Compute new weights
    d_new = 1 - np.power(g, q)
    
    # Update weights
    d = d_new * d_prev
    d = np.clip(d, d_limit, 1)
    d_ = 1/ (np.abs(Lx).flatten()**2+0.01**2)**0.25
    
    return d, g

def cr_rmmgks(
    A, b, x_true, L, nx, ny, delta,
    N=20, n_iter_first=100, n_iter_rest=100,
    q=2.0, pnorm=2, qnorm=1, kmin=5, kmax=20,
    regparam='dp', non_neg=True,epsilon=1e-2, tolambdah=1e-6, 
    break_lambdah=False, break_x=False, break_check=0, break_tol=1e-3,
    x_conv_tol=1e-4, d_limit=1e-3,use_non_neg=False, new_weights=True, adaptive_epsilon=False, max_total_iters=None,x0=None, V0=None, l0= None
):
    """
    Performs iterative reweighted image reconstruction.
    
    Returns:
        A dictionary containing the results and metrics of the reconstruction.
    """
    start_time = time.perf_counter()

    # --- Initialization ---
    if x0 is not None:
        x_t = x0.copy()
    else:
        x_t = np.ones((nx * ny, 1))
    x_old = np.zeros_like(x_t)
    V_t = V0
    l_t = l0 if l0 is not None else 0
    
    d = np.ones(L.shape[0])
    D = sparse.diags(d)

    # --- History Tracking ---
    history = {
        'rre': [],
        'residual_norm': [],
        'regularization_ratio': [],
        'weight_sum': [],
        'weights_d': [],
        'outer_iteration_infos': [],
        'x': []
    }

    print("--- Starting Iterative Reconstruction ---")
    all_rre = []
    all_lambda = []
    all_x = []
    for i in range(N):
        iter_start_time = time.perf_counter()
        
        if i > 0 and new_weights:
            d_prev = d.copy()
            Lx = L @ x_t
            d, g = compute_weights(Lx, D, d_prev, q, d_limit)
            D = sparse.diags(d)

        n_iter = n_iter_first if i == 0 else n_iter_rest
        
        # --- Core Reconstruction Step (assumes RMMGKS is defined) ---
        x_t, info_t, V_t, l_t = RMMGKS(
            A, b.flatten(order='F').reshape((-1, 1)), D @ L,
            pnorm=pnorm, qnorm=qnorm, projection_dim=kmin,
            n_iter=n_iter, regparam=regparam, x_true=x_true.reshape((-1, 1)),
            epsilon=epsilon, delta=delta, non_neg=non_neg, power=0.5, 
            x0=x_t.copy(), V0=V_t, kmin=kmin, l_max=kmax, 
            l_curve_plot=False, compute_V=False, tqdm_=False, lambdah=l_t, 
            tolambdah=tolambdah, break_lambdah=break_lambdah, 
            break_x=break_x, break_check=break_check, break_tol=break_tol, 
            use_non_neg=use_non_neg, adaptive_epsilon=adaptive_epsilon
        )
        history['x'].append(x_t.reshape(nx, ny))
       
        # --- Metrics and Logging ---
        current_rre = info_t['relError'][-1]
        current_lambda = info_t['regParam']
        all_lambda.extend(info_t['regParam_history'])
        all_rre.extend(info_t['relError'])
        all_x.extend(info_t['xHistory'])
        history['rre'].append(current_rre)
        history['all_rre'] = all_rre
        history['all_lambda'] = all_lambda
        history['all_x'] = all_x
        history['residual_norm'].append(la.norm(A @ x_t - b))
        #history['regularization_ratio'].append(la.norm(D @ L @ x_t) / la.norm(L @ x_t))
        #history['weight_sum'].append(np.sum(d))
        history['weights_d'].append(d.copy())
        #history['outer_iteration_infos'].append(info_t)
        
        iter_time = time.perf_counter() - iter_start_time
        print(f"Outer Iteration {i+1}/{N}: RRE = {current_rre:.4f}, Time = {iter_time:.2f}s")

        # --- Convergence Check for x ---
        diff_norm = la.norm(x_t - x_old)
        x_norm = la.norm(x_t)
        if x_norm > 0 and (diff_norm / x_norm) < x_conv_tol:
            print(f"\nConvergence reached at iteration {i+1}: Change in x is below tolerance.")
            break
        x_old = x_t.copy()

        if max_total_iters is not None and len(all_rre) >= max_total_iters:
            print(f"\nReached maximum total iterations ({max_total_iters}). Stopping.")
            break

    # --- Finalize and Return Results ---
    total_time = time.perf_counter() - start_time
    total_inner_iters = len(all_rre) #np.sum([len(info['regParam_history']) for info in history['outer_iteration_infos']])
    
    print(f"\n--- Reconstruction Finished ---")
    print(f"Total Outer Iterations: {i + 1}")
    print(f"Total Inner Iterations: {total_inner_iters}")
    print(f"Total Execution Time: {total_time:.2f}s")
    
    results = {
        'reconstructed_image': x_t.reshape(nx, ny),
        'true_image': x_true,
        'total_time': total_time,
        'total_outer_iterations': i + 1,
        'total_inner_iterations': total_inner_iters,
        'history': history,
        'parameters': {'N': N, 'q': q, 'pnorm': pnorm, 'qnorm': qnorm, 'kmin': kmin, 'kmax': kmax, 'regparam': regparam}
    }
    return results

def plot_reconstruction_results(results, nx, ny,num_plots=5):
    """
    Visualizes the results from the iterative reconstruction.
    """
    history = results['history']
    params = results['parameters']
    n_outer = results['total_outer_iterations']
    
    if n_outer == 0:
        print("No iterations were completed. Cannot plot results.")
        return

    plot_indices = np.linspace(0, n_outer - 1, num=min(n_outer, num_plots), dtype=int)
    
    fig, ax = plt.subplots(2, len(plot_indices), figsize=(20, 4), squeeze=False)
    
    for j, i in enumerate(plot_indices):
        #info_t = history['outer_iteration_infos'][i]
        d = history['weights_d'][i]
        x_recon_at_iter = history['x'][i]

        # Plot 1: Weights d
        ax[0, j].plot(d)
        ax[0, j].set_title(f'Iteration {i+1}')
        ax[0, j].set_ylim(0, 1.1)

        # Plot 2: Inner loop RRE
        # ax[1, j].plot(info_t['relError'])
        # ax[1, j].set_yscale('log')

        # Plot 3: Reconstructed image/signal
        if ny == 1: # 1D signal
            ax[1, j].plot(results['true_image'], label='True')
            ax[1, j].plot(x_recon_at_iter, '--', label='Recon')
            ax[1, j].legend()
        else: # 2D image
            ax[1, j].imshow(x_recon_at_iter.reshape(nx, ny), cmap='gray')
        
        # Plot 4: Regularization parameter history
        # ax[3, j].plot(info_t['regParam_history'])
        # ax[3, j].set_yscale('log')

    row_titles = ["Weights (d)", "Reconstruction"]
    for i, title in enumerate(row_titles):
        ax[i, 0].set_ylabel(title, fontsize=12)
        
    fig.suptitle(f'Reconstruction Progress (q={params["q"]}, pnorm={params["pnorm"]},qnorm = {params["qnorm"]}, kmin={params["kmin"]})', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    ax[0].plot(history['all_rre'])
    ax[0].set_title('Outer Loop RRE')
    ax[0].set_xlabel('Outer Iteration')
    ax[0].set_ylabel('RRE')
    ax[0].grid(True)
    ax[1].plot(history['all_lambda'])
    ax[1].set_title('Outer Loop λ')
    ax[1].set_xlabel('Outer Iteration')
    ax[1].set_ylabel('λ')
    ax[1].grid(True)
    row_titles = ["RRE history", "λ History"]
