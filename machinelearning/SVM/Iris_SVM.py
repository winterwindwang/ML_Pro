from sklearn import datasets
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 设置random.seed
np.random.seed(7)
tf.set_random_seed(7)

iris = datasets.load_iris()
X_data = np.array([[x[0], x[3]] for x in iris.data])
y_data = np.array([1 if y == 0 else -1 for y in iris.target])

# 重组数据集
X_data, y_data = shuffle(X_data, y_data, random_state=5)
# 分离测试集和训练集
train_indices = np.random.choice(len(X_data),
                                 round(len(X_data) * 0.8),
                                 replace=False)
test_indices = np.array(list(set(range(len(X_data))) - set(train_indices)))
X_train = X_data[train_indices]
X_test = X_data[test_indices]
y_train = y_data[train_indices]
y_test = y_data[test_indices]

# 定义模型和损失函数
batch_size = 135

# 定义输入
xs = tf.placeholder(shape=[None, 2], dtype=tf.float32)
ys = tf.placeholder(shape=[None, 1], dtype=tf.float32)

W = tf.Variable(tf.random_normal(shape=[2,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
# 定义线性模型
model_output = tf.subtract(tf.matmul(xs, W) , b)

l2_norm = tf.reduce_sum(tf.square(W))

alpha = tf.constant([0.01])
classification = tf.reduce_mean(tf.maximum(0.,tf.subtract(1., tf.multiply(model_output, ys))))
loss = tf.add(classification, tf.multiply(alpha, l2_norm))

prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, ys),tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_vec = []
train_accuracy = []
test_accuracy = []

for i in range(2000):
    rand_index = np.random.choice(len(X_train), size=batch_size)
    rand_x = X_train[rand_index]
    rand_y = np.transpose([y_train[rand_index]])
    sess.run(train_step, feed_dict={xs: rand_x, ys: rand_y})
    temp_loss = sess.run(loss, feed_dict={xs: rand_x, ys: rand_y})
    loss_vec.append(temp_loss)

    train_accuracy_temp = sess.run(accuracy, feed_dict={xs:X_train,ys:np.transpose([y_train])})
    train_accuracy.append(train_accuracy_temp)

    test_accuracy_temp = sess.run(accuracy, feed_dict={xs:X_test, ys:np.transpose([y_test])})
    test_accuracy.append(test_accuracy_temp)
    if (i + 1) % 100 == 0:
        # print('Step #{} A = {}, b = {}'.format(
        #     str(i+1),
        #     str(sess.run(W)),
        #     str(sess.run(b))
        # ))
        print('Accuracy = ' + str(test_accuracy_temp))

[[a1], [a2]] = sess.run(W)
[[b]] = sess.run(b)
slop = -a2 / a1
y_intercept = b / a1
best_fit = []

x1_data = [d[1] for d in X_data]

for i in x1_data:
    best_fit.append(slop*i + y_intercept)

setosa_x = [d[1] for i, d in enumerate(X_data) if y_data[i] == 1]
setosa_y = [d[0] for i, d in enumerate(X_data) if y_data[i] == 1]
not_setosa_x = [d[1] for i, d in enumerate(X_data) if y_data[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(X_data) if y_data[i] == -1]

plt.figure()
plt.plot(setosa_x, setosa_y , 'o', label='I-setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_data, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0,10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()

# plot train / test accuracies
plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per  Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()