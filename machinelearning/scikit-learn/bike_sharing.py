import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from some_func import plot_feature_importances
from sklearn.utils import shuffle


def load_datset(filename):
    file_reader = csv.reader(open(filename,'r'),  delimiter=',')
    X, y =[], []
    for row in file_reader:
        X.append(row[2:14])
        y.append(row[-1])
    # 提取特征名
    feature_names = np.array(X[0])

    # 将第一列名称移除，只保留数值
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names
# 读取数据
filename = r'D:\student\神经网络\数据集\共享单车数据集\Bike-Sharing-Dataset\day.csv'
X, y, feature_names = load_datset(filename)
X, y = shuffle(X, y, random_state=7)
# 训练和测试样本
num_training = int(0.9 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 训练回归器  参数n_estimators是指评估器（estimator）的数量，表示随机森林需要使用的决策
# 树数量；参数max_depth是指每个决策树的最大深度；参数min_samples_split是指决策树分
# 裂一个节点需要用到的最小数据样本量
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10)
rf_regressor.fit(X_train,y_train)

# 评价随机森林回归器的训练效果
y_pred_rd = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_rd)
evs = explained_variance_score(y_test, y_pred_rd)
print('\n###随机森林学习效果####')
print('Mean squared error = ',round(mse,2))
print('Explained variance error = ',round(evs,2))

# 画出特征重要性条形图
plot_feature_importances(rf_regressor.feature_importances_,'Random_tree regressor', feature_names)