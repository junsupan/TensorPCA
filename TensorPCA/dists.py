#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For creating and storing distributions

do not run this file
"""

import numpy as np
# from TensorPCA.hyptest import dist
from hyptest import dist

for k in range(1,10):
    for K in range(k+1,11):
        print('k='+str(k)+', K='+str(K))
        _, _, TW_dist = dist(k,K)
        TW_dist.tofile('k'+str(k)+'K'+str(K)+'.csv', sep = ',')
        del TW_dist
        
# np.loadtxt('distributions/a.csv',delimiter=',')