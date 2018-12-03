import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from Clustering_libtsp import kmeans,showCluster

print('step 1: load test data')
with open('libstp.tsp') as file:
    node_coord_start = None
    dimension = None
    lines = file.readlines()
    i = 0

    print(len(lines))
    while not dimension or not node_coord_start:
        line = lines[i]
        if line.startswith('DIMENSION'):
            dimension = int(line.split()[-1])
        if line.startswith('NODE_COORD_SECTION'):
            node_coord_start = i
        i = i + 1
    print('done')
    file.seek(0)
    dataSet = pd.read_csv(
        file,
        skiprows=node_coord_start+1,
        sep=' ',
        names={'y','x'},
        dtype={'y':np.float,'x':np.float},
        header = None,
        nrows = dimension
    )
    print(dataSet.shape)
    # for line in file.readlines():
    #     lineArr = line.strip().split('\t')
    #     dataSet.append([float(lineArr[0]),float(lineArr[1])])

print('step 2:clustering')
dataSet = np.mat(dataSet)
k = 3
centroid, clusterAssment = kmeans(dataSet,k)

print('step 3:show the result')
showCluster(dataSet,k ,centroid,clusterAssment)