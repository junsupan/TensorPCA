#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 12:53:29 2023

@author: junsupan
"""

import numpy as np
from numpy import linalg as LA
from scipy.stats import ortho_group
from .base import factor2tensor

def DGP(shape, R):
    """
    Generates the tensor data as a strong factor model introduced in the paper.
    The first dimension is treated as temporal, and generates AR(1) process factors.

    Parameters
    ----------
    shape : tupe
        shape of the tensor.
    R : int
        rank or number of factors.

    Returns
    -------
    Tensor.

    """
    d = len(shape)
    
    T = shape[0]
    
    # generating factor
    rho = 0.5
    sig_e = 0.1
    F = np.empty((T+100,R))
    e = sig_e * np.random.normal(0,1,(T+100,R))
    F[0,:] = e[0,:]
    for t in range(1,T+100):
        F[t,:] = F[t-1,:] * rho + e[t,:]
    F = F[100:,:]
    for r in range(R):
        F[:,r] = F[:,r] / LA.norm(F[:,r])
        
    # generating loadings
    M = [F]
    for j in range(1,d):
        M.append(ortho_group.rvs(shape[j])[:,0:R])
        
    # generating scale component, singal strength
    s = np.sqrt(np.prod(shape)) * np.array(range(R,0,-1))
    # s = np.sqrt(np.prod(shape)) * np.array([2,0.8])
    
    Y = factor2tensor(s,M)
    
    # generating idiosyncratic noise
    sig_u = 1
    U = sig_u * np.random.normal(0,1,shape)
    
    # add noise to tensor Y
    Y = Y + U
    
    return Y, s, M