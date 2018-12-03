import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# number one to ten data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this year
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784]) # 28*28 = 784
ys = tf.placeholder(tf.float32,[None,10])

# add output layer
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)
# [0.1,0.001,0.6,..] = [0,0,1,..]

#the error between predication and real data
cross_enptroy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1])) # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_enptroy)

sess = tf.Session()

# important step
sess.run(tf.global_variables_initializer())

for i in range(1000):
    xs_batch,ys_batch = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:xs_batch,ys:ys_batch})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
