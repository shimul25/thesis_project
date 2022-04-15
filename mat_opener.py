
from scipy.io import savemat, loadmat


high = loadmat('low.mat')
data = high['data']

print(data.shape)