from scipy.io import savemat, loadmat
import numpy as np

high = loadmat('C:/Users/yamraj/PycharmProjects/All_Matrices/high.mat')
dataA = high['data']
print("DataA Shape : ")
print(dataA.shape)

dataa = []

for i in range(42):
  temp = (dataA[i].flatten())
  temp = np.append(temp, np.zeros(5364))
  dataa.append(temp.reshape(448,448,3))


dataa = np.array(dataa)
# dataa = dataa.reshape(42, 448, 448, 3)
dataA = dataa
print("After appending zeroes DataA Shape : ")
print(dataA.shape)


low = loadmat('C:/Users/yamraj/PycharmProjects/All_Matrices/low.mat')
dataB = low['data']
print("DataB Shape : ")
print(dataB.shape)

datab = []

for i in range(42):
  temp = (dataB[i].flatten())
  temp = np.append(temp, np.zeros(5364))
  datab.append(temp.reshape(448,448,3))


datab = np.array(dataa)
# dataa = dataa.reshape(42, 448, 448, 3)
dataB = datab
print("After appending zeroes DataB Shape : ")
print(dataB.shape)

print("---------------------------------")
print("plot source matrix\n")
for dataa in dataA:
  print(dataa)
  
print("---------------------------------")  
print("plot target matrix\n")
for dataa in dataB:
  print(dataa)  