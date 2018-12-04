import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
import sklearn.metrics as sm

# 读取数据
input_file = r'D:\student\神经网络\数据集\交通量\traffic_data.txt'
X = []
count =0
with open(input_file,'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)
X = np.array(X)

# 将字符串转成数值
label_encoder = []
X_encoder = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoder[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoder[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoder[:, :-1].astype(int)
y = X_encoder[:, -1].astype(int)

# 建立SVR
params = {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.2,'gamma':'auto'}  # epsilon指定了不适用惩罚的限制
regressor = SVR(**params)
regressor.fit(X, y)


# 交叉验证
y_pred= regressor.predict(X)
print('Mean absoult error = ',round(sm.mean_absolute_error(y, y_pred), 2))

# 对单一数据示例进行编码测试
input_data = ['Tuesday', '13:35', 'San Francisco', 'yes']
input_data_encoded = [-1]*len(input_data)

count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = input_data[i]
    else:
        labels = []
        labels.append(input_data[i])
        input_data_encoded[i] = int(label_encoder[count].transform(labels))
        count = count + 1
input_data_encoded = np.array(input_data_encoded)
# 为特定数据点测试并打印分类结果
print('Predict traffic: ', int(regressor.predict(input_data_encoded.reshape(1, -1))[0]))
