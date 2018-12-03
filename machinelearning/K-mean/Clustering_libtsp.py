# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


def euclDistance(vector1, vector2):
    # cacluate Eucliden distance
    return np.sqrt(np.sum(np.power(vector2 - vector1,2)))

# 根据数据集设置指定k的质心
def initCentroids(dataSet, k):
    numSamples ,dim = dataSet.shape
    centroid = np.zeros((k,dim))
    for i in range(k):
        index = int(np.random.uniform(0,numSamples))
        centroid[i, : ] = dataSet[i, : ]
    return centroid

def kmeans(dataset, k):
    numSamples = dataset.shape[0]
    clusterAssment = np.mat(np.zeros((numSamples,2)))
    clusterChanged = True
    # k个集的质心
    centroids = initCentroids(dataset,k)

    while clusterChanged:
        clusterChanged = False
        for i in range(numSamples):
            miniDist = 100000.0
            miniIndex = 0

            for j in range(k):
                # calcuate the distance between sample and centroid
                distance = euclDistance(centroids[j,:],dataset[i,:])
                if distance < miniDist:
                    miniDist = distance
                    miniIndex = j
            # 质心是否更新
            if clusterAssment[i, 0] != miniIndex:
                clusterChanged = True
                clusterAssment[i,:] = miniIndex, miniDist**2

        for j in range(k):
            pointsInCluster = dataset[np.nonzero(clusterAssment[:,0 ].A ==j)[0]]
            centroids[j,:] = np.mean(pointsInCluster, axis=0)
        return centroids, clusterAssment
def showCluster(dataSet, k, centroids, clusterAssment):
    numSample , dim = dataSet.shape
    if dim != 2:
        print('请输入二维数据!')
        return 1
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("输入的k值有误！")
        return 1

    for i in range(numSample):
        markIndex = np.int(clusterAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i,0], centroids[i,1], mark[i],markersize=12)
    plt.show()