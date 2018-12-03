from sklearn.naive_bayes import  GaussianNB
from some_func import plot_classifier
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
input_file = r'D:\student\神经网络\数据集\小麦种子数据集\Wheat Seeds .txt'

X = []
y = []

with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split('\t')]
        X.append(data[:-1])
        y.append(data[-1])

X = np.array(X)
y = np.array(y)
X, y = shuffle(X, y, random_state=7)

# print('shape of X  = ',X.shape)
# print('shape of y  = ',y)

# 建立朴素贝叶斯分类器
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)
accuracy = 100.0 * ( y == y_pred).sum() / X.shape[0]
print('Accuracy of the classifier = ',round(accuracy, 2),"%")

# 数据特征维度大于2
# plot_classifier(classifier_gaussiannb, X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=5)
classifier_gaussiannb_new = GaussianNB()
classifier_gaussiannb_new.fit(X_train, y_train)
y_pred_new = classifier_gaussiannb_new.predict(X_test)

accuracy = 100.0 * (y_test == y_pred_new).sum() / X_test.shape[0]
print('Accuracy of the classifier = ',round(accuracy, 2),"%")

# 计算精度指标
num_validations = 5
accuracy = cross_val_score(classifier_gaussiannb, X, y, scoring='accuracy', cv=num_validations)
print('Accuracy : '+ str(round(100*accuracy.mean(), 2)) + "%")

f1 = cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted',cv=num_validations)
print('F1 : ' + str(round(100*f1.mean(), 2)) + "%")

precision = cross_val_score(classifier_gaussiannb, X, y, scoring='precision_weighted', cv=num_validations)
print('Precision : ' + str(round(100 * precision.mean(), 2)) + '%')

recall = cross_val_score(classifier_gaussiannb, X, y, scoring='recall_weighted', cv=num_validations)
print('Recall : ' + str(round(100 * recall.mean(), 2)) + "%")
