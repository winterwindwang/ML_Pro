import matplotlib.pyplot as plt
import numpy as np

def plot_classifier(classifier, X, y):
    # 定义图形的取值范围
    x_min, x_max = min(X[:, 0]) - 1.0 ,max(X[:, 0]) + 1.0
    y_min ,y_max = min(X[:, 1]) -1.0 ,max(X[:, 1]) + 1.0
    # 设置网格数的步长
    step_size = 0.01
    
    #定义网格
    x_value, y_value = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # 计算分类器输出结果
    mesh_ouput = classifier.predict(np.c_[x_value.ravel(),y_value.ravel()])

    # 数组维度变形
    mesh_ouput = mesh_ouput.reshape(x_value.shape)

    # 用彩色图划分楚结果
    plt.figure()
    # 选择配色方案
    plt.pcolormesh(x_value, y_value, mesh_ouput, cmap=plt.cm.gray)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    # 设置图形的取值范围
    plt.xlim(x_value.min(), x_value.max())
    plt.ylim(y_value.min(), y_value.max())

    # 设置X轴和y轴
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))

    plt.show()


def plot_feature_importances(feature_importances, title, feature_names):
    # 将重要性值标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # 将得分从高到低排序
    index_sorted = np.flipud(np.argsort(feature_importances))

    # 让X轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # 画条形图
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted], fontsize=10)
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()