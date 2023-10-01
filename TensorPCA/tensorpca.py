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
            s, gamma = LA.eig(self.unfolded[str(mode)] @ self.unfolded[str(mode)].transpose()) # Eigen-decomposition
            self.s_hat[str(mode)] = np.sqrt(np.sort(s)[::-1][:R]).real # scale components
            self.m_hat[str(mode)] = gamma[:,s.argsort()[::-1][:R]].real # vector components
            
        return self.s_hat, self.m_hat
            
    
    def ranktest(self, k, K, M=5000, progress=True):
        """
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

        
        if progress == True:
            print('Approximating Statistic Distribution, Progress:')
        elif progress == False:
            print('Statistic Distribution Approximating Progress Hidden.')
        
        # Simulates the TW distribution for the test statistic
        self.TW_dist = np.empty(M)
        for i in range(M):
            
            if (i+1) % 100 == 0:
                print(str(i+1)+'/'+str(M)) # displaying progress
                
            Z = np.random.normal(0,1,(1000,1000))
            Z = np.tril(Z,-1)
            Z = Z + Z.transpose()
            Z[np.diag_indices_from(Z)] = np.random.normal(0,np.sqrt(2),1000)
            
            s, _ = LA.eig(Z)
            s = np.sort(s)[::-1]
            
            eig_ratio = np.empty(K-k)
            for r in range(K-k):
                eig_ratio[r] = (s[r] - s[r+1])/(s[r+1] - s[r+2])
            
            self.TW_dist[i] = max(eig_ratio)
            
        # Test for each dimension
        self.S = np.empty(self.order)
        self.p = np.empty(self.order)
        for mode in range(self.order):
            # Calculates eigen value for each dimension
            s, _ = LA.eig(self.unfolded[str(mode)] @ self.unfolded[str(mode)].transpose())
            s = np.sort(s)[::-1]
            
            # Calculates the test statistic
            eig_ratio = np.empty(K-k)
            for r in range(K-k):
                eig_ratio[r] = (s[r] - s[r+1])/(s[r+1] - s[r+2])
            
            self.S[mode] = max(eig_ratio)
            self.p[mode] = sum(self.TW_dist > self.S[mode])/M
            
        return self.S, self.p



    
    
    
    
        