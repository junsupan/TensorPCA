#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 21:38:57 2023

@author: junsupan
"""

from setuptools import setup

setup(
    name='TensorPCA',
    version='0.1.0',    
    description='Tensor Principal Component Analysis',
    url='https://github.com/shuds13/pyexample',
    author='Junsu Pan',
    author_email='junsupan1994@gmail.com',
    license='GNU General Public',
    packages=['TensorPCA'],
    install_requires=['scipy',
                      'numpy'                     
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU License',  
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)