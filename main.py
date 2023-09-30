#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 11:46:16 2023

@author: junsupan
"""
import numpy as np

from Tensor.tensorpca import TensorPCA
from Tensor.dgp import DGP


R = 2 # rank
# tensor size TxNxJ
T = 100
N = 30
J = 20


Y, s, M = DGP((T,N,J),R)

Z = TensorPCA(Y)

s_hat, M_hat = Z.t_pca(2)