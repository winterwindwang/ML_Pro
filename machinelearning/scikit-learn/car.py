from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

input_file = r'D:\student\神经网络\数据集\汽车数据集\car_data.txt'

# 读取数据
X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')     # 忽略每行的最后一个字符，因为它是换行符
        X.append(data)

X = np.array(X)

# 将字符串转化为数值
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:,i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# 建立随机深林分类器
param = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}
classifier = RandomForestClassifier(**param)
classifier.fit(X, y)

# 交叉验证
accuracy = cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
print('Accuracy of the classifier: ' + str(round(100 * accuracy.mean(), 2)) + '%')

# 对单一数据示例进行编码测试
input_data = ['vhigh', 'vhigh', '2', '2', 'small', 'low']
input_data_encoded = [-1] * len(input_data)

for i, item in enumerate(input_data):
    label = []
    label.append(input_data[i])
    input_data_encoded[i] = int(label_encoder[i].transform(label))

input_data_encoded = np.array(input_data_encoded).reshape(1,-1)

# 预测并打印特定数据点的输出
output_class = classifier.predict(input_data_encoded)
print('Output class:', label_encoder[-1].inverse_transform(output_class)[0])

# 生成验证曲线
classifier = RandomForestClassifier(max_depth=4, random_state=7)
parameter_grid = np.linspace(25, 200, 8).astype(int)
train_scores, validation_scores = validation_curve(classifier, X, y, 'n_estimators', parameter_grid,cv=5)
print('\n####VALIDATION CURVE####')
print('\nParam: n_estimators\nTraining scores:\n', train_scores)
print('\nParam: n_estimators\nValidation scores:\n', validation_scores)

# 画出图形
plt.figure()
plt.plot(parameter_grid, 100 * np.average(train_scores, axis=1), color='black')
plt.title('Training curve')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()

classifier = RandomForestClassifier(n_estimators=20, random_state=7)
parameter_grid = np.linspace(2, 10, 5).astype(int)
train_scores, validation_scores = validation_curve(classifier, X, y, 'max_depth', parameter_grid, cv=5)
print('\n####VALIDATION CURVE####')
print('\nParam: n_estimators\nTraining scores:\n', train_scores)
print('\nParam: n_estimators\nValidation scores:\n', validation_scores)

plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Validation curve')
plt.xlabel('Maxinum depth of the type')
plt.ylabel('Accuracy')
plt.show()

# 生成学习曲线
classifier = RandomForestClassifier(random_state=7)
parameter_grid = np.array([200, 500, 800, 1100])
train_size, train_scores, validation_scores = learning_curve(classifier, X, y, train_sizes=parameter_grid, cv=5)
print('\n####Learning CURVE####')
print('\nParam: n_estimators\nTraining scores:\n', train_scores)
print('\nParam: n_estimators\nValidation scores:\n', validation_scores)

plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Learning curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()
