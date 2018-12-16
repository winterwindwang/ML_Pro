import tensorflow as tf
import numpy as np

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

batch_size = 8

# 1*2的矩阵 输入形状(n,2)表示有2个输入的n个训练样本
x = tf.placeholder(tf.float32, shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1),name='y-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

crosss_entropy = tf.reduce_mean(y * tf.log(tf.clip_by_value(y, 1e-10,1.0)))
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(crosss_entropy)

rdm = np.random.RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1,x2) in X]

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(train_step,feed_dict={x:[[0.7, 0.9],[0.1, 0.4], [0.5, 0.8]]}))
    print(sess.run(w1))
    print(sess.run(w2))
    STEPS= 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(crosss_entropy, feed_dict={x:X,y_:Y})
            print('After %d training step(s) ,cross entropy on all data is %g'%(i,total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
