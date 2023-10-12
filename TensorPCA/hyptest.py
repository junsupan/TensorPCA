#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:42:29 2023

@author: junsupan
"""

import pkg_resources
import numpy as np
from numpy import linalg as LA

def dist(k, K, M=5000, progress=True):
    """
    Approximates distribution of the statistic under the follow hypothesis
    Hypothesis Testing:
        Null: rank <= k
        Alternative: k < rank <= K
    where rank means the number of factors

    Parameters
    ----------
    k : int
        number of factors to be tested, "k" in the paper.
    K : int
        upper limit of the alternative hypothesis.
    M : int, optional
        number of repitions to approximate the statistic distribution. The default is 5000.
    progress : boolean, optional
        whether to display the progress of approximation. The default is True.

    Returns
    -------
    array
        Test statistics in each mode (dimension).
    array
        p-values of the statistics in each mode.

    """
    if k >= K:
        raise ValueError('k must be smaller than K')
        
    if k <= 0:
        raise ValueError('k must be a positive integer')
        
    # Using pre-created distributions
    if k in range(1,10):
        if K in range(k+1,11):
            path = pkg_resources.resource_filename(__name__, 'distributions/k'+str(k)+'K'+str(K)+'.csv')
            TW_dist = np.loadtxt(path,delimiter=',')
            return k, K, TW_dist
    
    if progress == True:
        print('The required hypothesis test \"k = '+str(k)+', K = '+str(K)+'\" does not have a pre-created distribution, initiating distribution approximation.\nProgress:')
    
    # Simulates the TW distribution for the test statistic
    TW_dist = np.empty(M)
    for i in range(M):
        
        if progress == True and (i+1) % 100 == 0:
            print(str(i+1)+'/'+str(M)) # displaying progress
            
        Z = np.random.normal(0,1,(1000,1000))
        Z = np.tril(Z,-1)
        Z = Z + Z.transpose()
        Z[np.diag_indices_from(Z)] = np.random.normal(0,np.sqrt(2),1000)
        
        s, _ = LA.eigh(Z)
        s = np.sort(s)[::-1]
        
        eig_ratio = np.empty(K-k)
        for r in range(K-k):
            eig_ratio[r] = (s[r] - s[r+1])/(s[r+1] - s[r+2])
        
        TW_dist[i] = max(eig_ratio)
        
    return k, K, TW_dist