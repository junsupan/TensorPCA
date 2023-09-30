#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:38:06 2023

@author: junsupan
"""
import numpy as np
from scipy.linalg import khatri_rao


def fold(Matrix, shape):
    """
    

    Parameters
    ----------
    Matrix : array_like
        A matrix of unfolded tensor.
    shape : tuple of shape of the tensor

    Returns
    -------
    nd array tensor.

    """
    if np.size(Matrix) != np.prod(np.array(shape)):
        raise ValueError('The desired shape does not matched the given matrix.')
    
    return np.reshape(Matrix, shape, order='F')

def factor2tensor(s, M):
    """
    Takes tensor product of vector components and scale component of a tensor factor model.

    Parameters
    ----------
    s : list or 1d array of scale components.
    M : list of facotr/loading matrices, each matrix should be N_d x R with the same rank R.

    Returns
    -------
    A tensor.

    """
    
    R = len(s)
    
    d = len(M)
    
    N = []
    for i in range(d):
        N.append(M[i].shape[0])
        if M[i].shape[1] != R:
            raise ValueError('Facor/loading matrices should have the same number of columns.')
    
    # calculate the matricied tensor using equation (7) in the paper
    t = M[0]
    
    t = t @ np.diag(s)
    
    KR = M[-1]
    
    if d > 2:
        for j in range(2,d):
            KR = khatri_rao(KR, M[-j])
            
    t = t @ KR.transpose()
    
    t = fold(t, tuple(N))
    
    return t
        