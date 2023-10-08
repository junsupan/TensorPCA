#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MC simulation for selecting the number of factors

When R=1, we expect the rejection probability to be 0.05;
when R=2, we expect the rejection probability to be 1.

"""

import numpy as np

from TensorPCA.tensorpca import TensorPCA
from TensorPCA.dgp import DGP
from TensorPCA.hyptest import dist

R = 2 # rank
# tensor size TxNxJ
T = 40
N = 30
J = 20

# approximates distribution
TW_dist = dist(1, 3)

ps = np.empty((5000,3))

for i in range(5000):
    if (i+1) % 100 == 0:
        print(str(i+1)) # displaying progress
    
    # Generate a random tensor factor model and store the tensor
    Y, _, _ = DGP((T,N,J),R)
    
    # input the tensor into TensorPCA class
    Z = TensorPCA(Y)
    
    # rank test
    _, p = Z.ranktest(TW_dist)
    ps[i,:] = p
    
# rejection probability in each dimension
a=np.sum(ps<0.05,0)/5000
print('Empirical rejection probability in each dimension: '+str(a))