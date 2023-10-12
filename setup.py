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
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/junsupan/TensorPCA',
    author='Junsu Pan',
    author_email='junsupan1994@gmail.com',
    license='MIT',
    packages=['TensorPCA'],
    install_requires=['scipy',
                      'numpy'                     
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    include_package_data=True,
)