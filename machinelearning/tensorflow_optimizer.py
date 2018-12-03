import tensorflow as tf
import numpy as np

# Trainning
#  Optimizers
# class tf.train.optimizer

# save the file

# remember to define the same dtype and shape when storing
# W = tf.Variable([[1,2,3],[1,2,3]],dtype=tf.float32,name='Weights')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess,'my_net/save_net.ckpt')
#     print('Save to path', save_path)

# restore variables
# redefine the same shape and dtype for your variable
W = tf.Variable(tf.zeros(2,3),dtype=tf.float32,name='Weights')
b = tf.Variable(tf.zeros(1,3),dtype=tf.float32,name='biases')

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.save(sess,'my_net/save_net.ckpt')
    print('Weights',sess.run(W))
    print('biases',sess.run(b))