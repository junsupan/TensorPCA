#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:32:40 2023

@author: junsupan
"""

import numpy as np
from numpy import linalg as LA



class TensorPCA:
    
    def __init__(self, tensor):
        """
        Sets up initial parameters

        Parameters
        ----------
        tensor : array_like
            Tensor data.

        Returns
        -------
        None.

        """
        
        if np.isnan(tensor).any() == True:
            raise ValueError('The tensor contains missing values')
            
        self.tensor = tensor # store tensor itself
        self.order = tensor.ndim # order of the tensor
        self.shape = tensor.shape # shape of the tensor
        
        
        self.unfolded = {}
        for mode in range(self.order):
            self.unfolded[str(mode)] = np.moveaxis(self.tensor,mode,0) \
            .reshape((self.shape[mode],-1),order='F')
        
        
    def unfold(self, mode):
        """
        Returns unfolded tensor in jth mode

        Parameters
        ----------
        mode : int
            jth dimension of the tensor, j <= d.

        Returns
        -------
        array
            unfolded tensor.

        """

        return self.unfolded[str(mode)]
    
    
    def t_pca(self, R):
        """
        Calculates the tensor pca components

        Parameters
        ----------
        R : int
            number of factors.

        Returns
        -------
        dict
            estimated scale components for each mode.
        dict
            estimated vector components for each mode.

        """

        
        # Estimates Tensor PCA
        self.s_hat = {}
        self.m_hat = {}
        for mode in range(self.order):
            s, gamma = LA.eigh(self.unfolded[str(mode)] @ self.unfolded[str(mode)].transpose()) # Eigen-decomposition
            self.s_hat[str(mode)] = np.sqrt(np.sort(s)[::-1][:R]).real # scale components
            self.m_hat[str(mode)] = gamma[:,s.argsort()[::-1][:R]].real # vector components
            
        return self.s_hat, self.m_hat
            
    
    def ranktest(self, TW_dist):
        """
        Hypothesis Testing:
            Null: rank <= k
            Alternative: k < rank <= K
        where rank means the number of factors
        
        

        Parameters
        ----------
        TW_dist : tuple contains k, K, and the approximated distribution
            approximated distribution of statistic, run "dist" function first.

        Returns
        -------
        array
            Test statistics in each mode (dimension).
        array
            p-values of the statistics in each mode.

        """
        k = TW_dist[0]
        K = TW_dist[1]
        M = len(TW_dist[2])
        dist = TW_dist[2]
            
        # Test for each dimension
        self.S = np.empty(self.order)
        self.p = np.empty(self.order)
        for mode in range(self.order):
            # Calculates eigen value for each dimension
            s, _ = LA.eigh(self.unfolded[str(mode)] @ self.unfolded[str(mode)].transpose())
            s = np.sort(s)[::-1]
            
            # Calculates the test statistic
            eig_ratio = np.empty(K-k)
            for r in range(K-k):
                eig_ratio[r] = (s[k+r] - s[k+r+1])/(s[k+r+1] - s[k+r+2])
            
            self.S[mode] = max(eig_ratio)
            self.p[mode] = sum(dist > self.S[mode])/M
            
        return self.S, self.p



    
    
    
    
        