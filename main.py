import scipy.io as sio
from scipy.io import savemat
import torch


def makeSymmetric(mat):
    # Loop to traverse lower triangular
    # elements of the given matrix
    (m,n) = mat.shape
    for i in range(0, m):
        for j in range(0, n):
            if (j < i):
                mat[i][j] = mat[j][i] = (mat[i][j] +
                                         mat[j][i]) // 2




mat = sio.loadmat('dataA\C1090.mat')
print(mat)
f_mat = mat['chickenpiecesnorm10']

makeSymmetric(f_mat)

print(f_mat)

with torch.no_grad():
  mdic = {'data': [f_ for f_ in f_mat]}
  savemat('after_sym/c1090_.mat', mdic)

