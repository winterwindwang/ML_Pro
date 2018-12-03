import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# state = tf.Variable(0,name='counter')
# # print(state.name)
#
# one = tf.constant(1)
#
# new_value = tf.add(state , one)
# update = tf.assign(state,new_value)
#
# init = tf.global_variables_initializer() # must have if define variables
#
# with tf.Session() as sess:
#     # init 必须得run
#     sess.run(init)
#     for _ in range(3):
#         sess.run(update)
#         print(sess.run(state))

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]})) #python dictionary
