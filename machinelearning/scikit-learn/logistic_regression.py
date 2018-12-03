import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from some_func import plot_classifier

X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2],
[1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# 参数solver用于设置求解系统方程的算法类型，参数C表示正则化强度，数值越小，表示正则化强度越高
classifier = linear_model.LogisticRegression(solver='liblinear', C=1000)

classifier.fit(X, y)

plot_classifier(classifier, X, y)
