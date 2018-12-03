import sys
import os
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pickle

filename = os.getcwd() + '\data\data_singlevar.txt'
    # sys.argv[1]
x = []
y = []
i = 0
with open(filename) as f:
    for line in f.readlines():
        if i >=50:
            yt = np.float32(line)*10
            y.append(yt)
        else:
            xt = np.float32(line)*10
            x.append(xt)
        i = i + 1
# print(np.shape(x))
num_training = int(0.8*len(x))
num_test = len(x) - num_training

# 训练数据
x_train = np.array(x[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

# 测试数据
x_test = np.array(x[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

# 创建线性回归对象
linear_regressor = linear_model.LogisticRegression()
# 用训练数据集训练模型
print(np.shape(x_train))
print(np.shape(y_train))
linear_regressor.fit(x_train.astype('int'), y_train.astype('int'))

y_train_pred = linear_regressor.predict(x_train)
plt.figure()
plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,y_train_pred,color='black', linewidth=4)
plt.title('Training data')
plt.show()

y_test_pred = linear_regressor.predict(x_test)
plt.scatter(x_test,y_test,color='green')
plt.plot(x_test,y_test_pred,color='black',linewidth=4)
plt.title('Test data')
plt.show()

# 计算回归的准确性
print('Mean absoulte error = ', round(sm.mean_absolute_error(y_test,y_test_pred), 2))
print('Mean squared error = ',round(sm.mean_squared_error(y_test,y_test_pred), 2))
print('Median absolute error = ',round(sm.median_absolute_error(y_test,y_test_pred), 2))
print('Explianed variance error =',round(sm.explained_variance_score(y_test,y_test_pred), 2))
print('R2 score = ',round(sm.r2_score(y_test,y_test_pred), 2))

# 保持模型数据
output_model_file = 'saved_model.pkl'
# read byte is important
with open(output_model_file,'wb') as f:
    pickle.dump(linear_regressor,f)

# 使用加载并使用它
with open(output_model_file,'rb') as f:
    model_linear = pickle.load(f)
y_test_pred_new = model_linear.predict(x_test)
print('\nNew mean absolute error = ',round(sm.mean_squared_error(y_test,y_test_pred_new),2))
