
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


from dp import *

def RMMGKS(A, b, L, pnorm=2, qnorm=1, projection_dim=3, n_iter=5, regparam='gcv', x0 = None, V0 = None, x_true=None, power = 1, 
tqdm_ = True,kmin=3,l_max = 5,l_curve_plot=False,compute_V = True, alpha = 1,lambdah=None,tolambdah=1e-6, break_lambdah = False, break_x = False, break_tol=1e-3,break_check=0,use_non_neg=False,adaptive_epsilon=False,**kwargs):

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    isoTV_option = kwargs['isoTV'] if ('isoTV' in kwargs) else False
    GS_option = kwargs['GS'] if ('GS' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    prob_dims = kwargs['prob_dims'] if ('prob_dims' in kwargs) else False
    non_neg = kwargs['non_neg'] if ('non_neg' in kwargs) else False
    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]
    #gcvtype= kwargs['gcvtype'] if ('gcvtype' in kwargs) else 'tikhonov'
    

    if V0 is None:
        #print(V0)
        (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)

    else:
        #print(V0.shape)
        V = V0
    
    V,_  = np.linalg.qr(V) # Ensure orthonormality of V
    x_history = []
    lambda_history = [] if lambdah is None else [lambdah]
    residual_history = []
    residual2_history = []

    e = 1
    if x0 is not None:
        x = x0
    else:
        x = A.T @ b # initialize x for reweighting

    x = x.reshape(-1,1)
    AV = A@V
    diff = l_max - kmin
    LV = L@V
    if (tqdm_ == True):
        range_ = tqdm(range(n_iter), desc='running MMGKS...')
    else:
        range_ = range(n_iter)
    x_orig = x
    y = V.T @ x
    if lambdah is None:
        lambdah = 0
    
    #V = None
    for ii in range_:
        

        v = A @x_orig - b
        u = L @ x_orig
        if adaptive_epsilon:
            epsilon = np.median(np.abs(u))
            #print(np.max(np.abs(u)))
            epsilon = 0.01*np.max(np.abs(u)) #1.5*calculate_mad(u) #np.percentile(np.abs(u),10) #max(epsilon, 1e-3)
            #print('epsilon', epsilon,'max u', np.max(np.abs(u)), 'median u', np.median(np.abs(u)), '25th percentile u', np.percentile(np.abs(u), 25), '75th percentile u', np.percentile(np.abs(u), 75))
        wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))
        wf = smoothed_holder_weights(v, epsilon=epsilon, p=pnorm).reshape((-1,1))        
 
        ra = wf * (A @x_orig - b)
        ra = A.T @ ra
        rb = wr * (L @ x_orig)
        rb = L.T @ rb
        r = ra + lambdah * rb
        r1 = r
        #if V is not None:
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)
        normed_r = r / la.norm(r) 
        vn = r / np.linalg.norm(r)
        
        V = np.column_stack((V, vn))
        Avn = A @ vn
        AV = np.column_stack((AV, Avn))

        Lvn = L@vn
        LV = np.column_stack((LV, Lvn))


        
        residual_history.append(la.norm(r))
        residual2_history.append(la.norm(r1))

        v = A @ x - b
        wf = (v**2 + epsilon**2)**(pnorm/2 - 1)
        AA = AV*(wf**power)
        (Q_A, R_A) = la.qr(AA, mode='economic') 
        #print(v.shape,L.shape,b.shape, A.shape)
        u = L @ x
        wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))
        #print('yes')
        LL = LV * (wr**power)
        (Q_L, R_L) = la.qr(LL, mode='economic') 
        if regparam == 'gcv':
            lambdah = generalized_crossvalidation_2(Q_A, R_A, R_L, (wf**power) *b, **kwargs)
        elif regparam == 'gcv_tol':
            lambdah = generalized_crossvalidation_tol(Q_A, R_A, R_L, (wf**power) *b, **kwargs)
        elif regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, (wf**power) *b, **kwargs)
        elif regparam == 'l_curve':
            #print(Q_A.shape,((wf**power)*b).shape )
            lambdah = l_curve(R_A, R_L,Q_A.T@ ((wf**power)*b),plot=l_curve_plot)
        else:
            lambdah = regparam
        # if (len(lambda_history)>0) and (lambdah < tolambdah): #1e-6):
        #     #print(lambdah,lambda_history[-1])
        #     lambdah = deepcopy(lambda_history[-1])  
      
        # if break_:
        #     if (ii > 2) and (lambdah>break_check ) and la.norm(lambdah - lambda_history[-1]) / (lambda_history[-1] + 1e-12) < break_tol:
        #         print(f"Stopping early at iteration {ii} due to small change in lambda: {lambdah}")
        #         break
       # 1. Store the previous value for comparison
        prev_lambda = lambda_history[-1] if lambda_history else None

        # 2. Get the new lambda from GCV/DP/etc.
        # (Your if/elif block here...)

        # 3. Check for natural convergence BEFORE manual correction
        if break_lambdah and prev_lambda is not None and ii > 2:
            rel_diff = abs(lambdah - prev_lambda) / (prev_lambda + 1e-12)
            if lambdah > break_check and rel_diff < break_tol:
                print(f"Stopping naturally: lambda stabilized at {lambdah}")
                break


        # 4. Apply safety threshold (Correction)
        if prev_lambda is not None and lambdah < tolambdah:
            lambdah = deepcopy(prev_lambda)         
        lambda_history.append(lambdah)
        y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), 
                        np.concatenate((Q_A.T@ ((wf**power)*b), np.zeros((R_L.shape[0],1)))),rcond=None)
#        print(y.shape)

        x = V @ y
        x_orig = V@y
        #L_ = gen_first_derivative_operator_2D(nx, ny)
        if break_x and ii>2:
            rel_diff_x = la.norm(x - x_history[-1]) / (la.norm(x_history[-1]) + 1e-12)
            if rel_diff_x < break_tol:
                print(f"Stopping naturally: x stabilized at iteration {ii}")
                break
        #print(y.shape)
        #print('rre',la.norm(x-x_true)/la.norm(x_true))
        V_last = V
        if (non_neg):
            x[x<0] = 0
        x_history.append(x)
        # if ii >= R_L.shape[0]:
        #     break
        if use_non_neg:
            x_orig = x
        #print('yes')
        #print(V.shape[1])
        #print(y)

        if ((V.shape[1] >= l_max) or compute_V): # or  (ii == (n_iter -1))): 
            #print('Y')
            #regparam = lambdah

            # # # Compute truncated SVD with k singular values
            _,_ ,Wt = np.linalg.svd(np.vstack((R_A, np.sqrt(lambdah) * R_L)))

            W = Wt.T
            W = W[:, :kmin-1]

            V_tilde = V[:,:]@W

            # idx = np.argsort(np.abs(y.flatten()))[-(kmin-1):] if kmin>1 else []#[:(kmin-1)]  # indices of top |y|
            # V_tilde = V[:, idx]                      # select columns
            # print(y.flatten()[idx], y.flatten())
            #y = np.array(y[idx])
            #print(la.norm(V_tilde@y[idx]-x_orig))
            #np.save('x_tilde.npy', V_tilde@y[idx])
            #print(V_tilde.shape, y.shape)
            #V_tilde, _ = np.linalg.qr(V_tilde)

            x_new =  x_orig - V_tilde @ (V_tilde.T @ x_orig)#V[:,np.argsort(np.abs(y.flatten()))[-6:]]@y[np.argsort(np.abs(y.flatten()))[-6:]] #
            x_new /= la.norm(x_new)
            V = np.column_stack((V_tilde, x_new))#V_tilde@y[idx])) #
            #assert np.linalg.norm( (V.T @ V) - np.eye(V.shape[1]) ) < 1e-10, "New basis is not a basis"
            V, _ = np.linalg.qr(V)

            AV = A@V #np.column_stack((AV, Avn))

            LV = L@V #np.column_stack((LV, Lvn))


        
    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relError': rre_history, 'Residual': residual_history, 'Residual2': residual2_history, 'its': ii,'V':V, 'V_last':V_last, 'y':y}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history,  'Residual2': residual2_history,'its': ii,'V':V, 'V_last':V_last, 'y':y}
    
    return (x, info,V, lambdah)