#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example code

See 'simulation.py' for a simulation in Hypothesis testing

"""
import numpy as np

from TensorPCA.tensorpca import TensorPCA
from TensorPCA.dgp import DGP



R = 2 # rank
# tensor size TxNxJ
T = 40
N = 30
J = 20

# Generate a random tensor factor model and store the tensor
Y, s, M = DGP((T,N,J),R)

# input the tensor into TensorPCA class
Z = TensorPCA(Y)

# estimate tensor factor model parameters
s_hat, M_hat = Z.t_pca(2)


print('True parameters simulated:\n')

print(' - scale components: \n'+str(s))

print(' - vector components \n'+\
      'factors \n'+str(M[0])+'\n \n'+\
          'loadings lambda \n'+str(M[1])+'\n \n'+\
              'loadings mu \n'+str(M[2])+'\n \n')
    
print('Estimated parameters:')

print(' - estimated scale components: \n from each dimension \n'+str(s_hat)+'\n')

print(' - estimated vector components: \n'+\
      'estimated factors \n'+str(M_hat['0'])+'\n \n'+\
          'estimated loadings lambda \n'+str(M_hat['1'])+'\n \n'+\
              'estimated loadings mu \n'+str(M_hat['2']))
    
