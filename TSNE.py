# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:12:02 2022

@author: yamraj
"""

import os
import scipy.io as sio
import numpy
from scipy.io import savemat, loadmat
import torch
from sklearn.manifold import TSNE

dir  = 'C:/Users/yamraj/PycharmProjects/All_Matrices/dataA/C545.mat'
#os.chdir(dir)

for file_name in os.listdir():
    mat = sio.loadmat(file_name)
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z=tsne.fit_transform(mat[list(mat.keys())[3]])
print(z)
    