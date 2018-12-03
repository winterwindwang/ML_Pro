import numpy as np
import sys

filename_in = r'D:\browserDownload\ex2Data\ex2x.dat'
filename_out = r'D:\browserDownload\ex2Data\ex2y.dat'
x = []
y = []
with open(filename_in) as f:
    for line in f.readlines():
        xt = float(line)
        x.append(xt)

with open(filename_in) as f:
    for line in f.readlines():
        yt = float(line)
        y.append(xt)
reslut = np.column_stack((x,y))
print(reslut.shape)

filename_result= r'D:/Python/ML_Pro/machinelearning/scikit-learn/data/singlevar.txt'
with open(filename_result,'w')as f:
    for i in range(reslut.shape[0]):
        strs = str(reslut[i][0])+','+str(reslut[i][1])+'\n'
        f.write(strs)