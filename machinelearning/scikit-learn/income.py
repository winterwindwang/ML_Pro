import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

input_file = r'D:\student\神经网络\数据集\美国人口普查收入数据\adult.data.txt'

# 读取数据
X = []
y = []
count_lessthan50k = 0
count_morethan50k = 0
num_image_threshold = 10000

with open(input_file,'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue
        line.strip()
        # 去除数据中每个字段前的空格
        data = [i.strip() for i in line[:-1].split(',')]
        # 使用两种类型相同的数据点
        if data[-1].lstrip() == '<=50K' and count_lessthan50k < num_image_threshold:
            X.append(data)
            count_lessthan50k = count_lessthan50k + 1
        elif data[-1].lstrip() == '>50K' and count_morethan50k < num_image_threshold:
            X.append(data)
            count_morethan50k = count_morethan50k + 1
        if count_lessthan50k >= num_image_threshold and count_morethan50k >= num_image_threshold:
            break
X = np.array(X)
print(X.shape)
# 将字符串转换为数据值数据  isdigit()函数判断一个属性是否是数据数据
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.strip().isdigit():
        X_encoded[:,i] = X[:,i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:,i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)


# 建立分类器
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
print(type(X))
# 交叉验证
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb.predict(X_test)

# 计算分类器的F1得分
f1 = cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted', cv=5)
print('F1 scored: ' + str(round(100*f1.mean(), 2)) + '%')

# 对单一数据进行编码测试  (字符串的每个字段前注意要加空格)
input_data = ['39', 'State-gov', '77516', 'Bachelors', '13', 'Never-married',
'Adm-clerical', 'Not-in-family', 'White', 'Male', '2174', '0', '40', 'United-States']
count = 0
input_data_encoded = [-1] * len(input_data)
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        labels = []
        labels.append(input_data[i])
        input_data_encoded[i] = int(label_encoder[count].transform(labels))
        count = count + 1
input_data_encoded = np.array(input_data_encoded).reshape(1,-1)
# 预测并打印特定数据点的输出结果
output_class = classifier_gaussiannb.predict(input_data_encoded)
print(label_encoder[-1].inverse_transform(output_class)[0])