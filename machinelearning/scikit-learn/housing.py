import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.utils import shuffle
from some_func import plot_feature_importances
# input data
housing_data = datasets.load_boston()
# 打乱数据，random_state控制如何打乱数据
X , y = shuffle(housing_data.data,housing_data.target,random_state=7)

# 80%作为训练样本，20%作为测试样本
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 训练
dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train,y_train)

# 拟合
ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=400, random_state=7)
ab_regressor.fit(X_train, y_train)

# 预测
y_pred_dt = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test,y_pred_dt)
evs = explained_variance_score(y_test,y_pred_dt)
print("\n######决策树学习效果######")
print('Mean squared error  = ',round(mse, 2))
print('Explain variance error = ',round(evs, 2))

y_pred_ad = ab_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_ad)
evs = explained_variance_score(y_test, y_pred_ad)
print('\n#####AdaBoost算法改善效果#####')
print('Mean squared error = ',round(mse, 2))
print('Explained variance error = ',round(evs, 2))



# 计算特征的重要性
plot_feature_importances(dt_regressor.feature_importances_,'Decision Tree regressor', housing_data.feature_names)
plot_feature_importances(ab_regressor.feature_importances_,'AdaBoost regressor', housing_data.feature_names)