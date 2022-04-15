import os
import scipy.io as sio
import numpy
from scipy.io import savemat, loadmat
import torch

dir  = os.path.abspath('C:/Users/yamraj/PycharmProjects/All_Matrices/dataA')
os.chdir(dir)


high = []

for file_name in os.listdir():
    mat = sio.loadmat(file_name)
    #print(mat)
    f_mat = mat[list(mat.keys())[3]]
    print(f_mat)
    f_mat_3d = []
    f_mat_3d.append(f_mat)
    f_mat_3d.append(f_mat)
    f_mat_3d.append(f_mat)
    print(f_mat_3d)

    temp = numpy.array(f_mat_3d)
    print(temp)
    temp = temp.reshape(( 446, 446, 3))
    high.append(temp)


with torch.no_grad():
  mdic = {'data': [f_ for f_ in high]}
  savemat('high.mat', mdic)






