import numpy as np
import matplotlib.pyplot as plt 
from sklearn。model_selection import train_test_split
from sklearn.svm import SVC
# import utilities

# 加载输入数据
input_file = r"D:\student\神经网络\数据集\data_multicar\data_multivar.txt"

# 加载文件中多变量数据
def load_data(input_file):
	X = []
	y = []
	with open(input_file,'r') as f:
		for line in f.readlines():
			data = [float(x) for x in line.split(",")]
			X.append(data[:-1])
			y.append(data[-1])
	X = np.array(X)
	y = np.array(y)
	return X, y

X, y = load_data(input_file)# utilities.load_data(input_file)
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0]) 
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1]) 

plt.figure()
plt.scatter(class_0[:,0],class_0[:,1],facecolors='black', edgecolors='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], facecolors='None', edgecolors='black',marker='s')
plt.title('Input data')
# plt.show()

# 分割数据集并使用SVM训练
X_train. X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
params = {'kernel':'linear'}
classifier=SVC(**params)
classifier.fit(X_train,y_train)
