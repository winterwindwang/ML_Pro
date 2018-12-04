import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
import utilities

data = utilities.load_data('D:\student\神经网络\数据集\SVM\data_multivar.txt')
num_cluster = 4

# print(utilities.load_data('D:\student\神经网络\数据集\data_multicar\data_multivar.txt'))



plt.scatter(data[:, 0], data[:, 1], marker='o', facecolors='none', edgecolors='k', s=30)
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0] ) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
# plt.show()

kmeans = KMeans(init='k-means++', n_clusters=num_cluster, n_init=10)
kmeans.fit(data)

# 设置网络数据的步长
step_size = 0.01
# 画出边界
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0] ) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
x_values, y_values = np.meshgrid(np.arange(x_min,x_max,step_size), np.arange(y_min, y_max, step_size))

# 预测网格中所有数据点的标记
predict_labels = kmeans.predict(np.c_[x_values.ravel(),y_values.ravel()])
# 画出结果
predict_labels = predict_labels.reshape(x_values.shape)
plt.figure()
plt.clf()
plt.imshow(predict_labels, interpolation='nearest', extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
           cmap=plt.cm.Paired, aspect='auto', origin='lower')

plt.scatter(data[:, 0], data[:, 1], marker='o', facecolors='none', edgecolors='k', s=30)

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, linewidths=3, color='k', zorder=10, facecolors='black')
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0] ) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
plt.title('Centroids and boundaries obtained using KMeans')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()