# Just disables the warning, doesn't enable AVX/FMA
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# ignore warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# test programme
# hello = tf.constant('Hello,TesnorFlow!')
# sess = tf.Session()
# print(sess.run(hello))
# sess.close()

plt.figure()

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

plt.scatter(x_data,y_data,s=25,alpha=.5)
plt.xlim(-2.0,2.0)
# plt.xticks(())
plt.xlim(-1.0,4.0)
# plt.yticks(())
plt.show()

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init) # Very important

for step in range(201):
    sess.run(train)
    if step % 20 ==0:
        print(step,sess.run(Weights),sess.run(biases))

sess.close()

