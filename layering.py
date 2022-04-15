
# (x, 446, 446, 3)
import numpy
import scipy.io as sio



mat = sio.loadmat('C545.mat')
f_mat = mat['chickenpiecesnorm560']

f_mat_3d = []
f_mat_3d.append(f_mat)
f_mat_3d.append(f_mat)
f_mat_3d.append(f_mat)

temp = numpy.array(f_mat_3d)
temp = temp.reshape((446,446,3))
print(temp.shape)



