import numpy as np

# sigmoid function
def sigmoid(x,deriv = False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def dsigmoid(x,deriv = False):
    if (deriv == True):
        return 1.0 - x**2
    return x * (1 - x)

# input dataset (输入数据集，形式为矩阵，每一行代表一个训练样本)
X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

# output dateset 输出数据集，形式为向量，每一行代表一个训练样本
y = np.array([[0,0,1,1]]).T

# seed random number to make calculation
# deterministic (just make a good practice)
np.random.seed(1)

# initialize weights randomly  with mean 0
# 第一层权值，突触0，连接l0层与l1层
syn0 =  2 *np.random.random((3,1))-1

for iter in range(10000):
    # forward propagation
    # 网络第一层，即网络输入层
    l0 = X
    # 网络第二层，即隐藏层
    l1 = sigmoid(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1
    # multiply how much we missed by the slope of the sigmoid at the value in l1
    l1_deleta = l1_error * sigmoid(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_deleta)
print("Out After training :")
print(l1)












