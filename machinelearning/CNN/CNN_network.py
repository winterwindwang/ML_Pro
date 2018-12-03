import tensorflow as tf
import numpy as np
import os
from NotMNIST import NotMNIST
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# number 0 to 10
# mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

mnist = NotMNIST()

# run tensorboard : tensorboard --logdir=D:\Python\ML_Pro\machinelearning\CNN\logs --host=127.0.0.1

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_production'):
            correct_production = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_production,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    tf.summary.scalar('accuracy',result)
    return result



def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name='w')

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name='b')

def conv2d(x,W):
    # stride [1,x_movement,y_movement,1]
    # must stride[0]=stride[3]=1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    # stride [1,x_movement,y_movement,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# define placeholder for  inputs to works
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32,[None,784],name='x_inputs') /255. # 28*28
    ys = tf.placeholder(tf.float32,[None,10],name='y_inputs')

keep_prob = tf.placeholder(tf.float32,name='prob')
with tf.name_scope('input_reshape'):
    x_image = tf.reshape(xs,[-1,28,28,1]) # [nshape,28,28,1]
    tf.summary.image('input',x_image,10)

########## 测试用 #################
# def conv_layer(inputs, channels_in, channels_out, name="conv"):
#     with tf.name_scope(name):
#         w = weight_variable([3,3,channels_in,channels_out])
#         b = bias_variable([channels_out])
#         conv = tf.nn.relu(conv2d(inputs, w ) + b)
#         return max_pool_2x2(conv)
#
# def fc_layer(inputs, channel_in, channel_out, name="fc",fun=None):
#     with tf.name_scope(name):
#         w = weight_variable([channel_in,channel_out])
#         b = bias_variable([channel_out])
#         if fun is not None:
#             return tf.nn.relu(tf.matmul(inputs, w) + b)
#         else:
#             return tf.nn.softmax(tf.matmul(inputs, w) + b)
#
# conv1=conv_layer(x_image,1,32,'conv1')
# conv2=conv_layer(conv1,32,64,'conv2')
#
# flattened = tf.reshape(conv2,[-1,7*7*64])
# fc1 = fc_layer(flattened,7*7*64,1024,"fc1")
# fc1_drop_out = tf.nn.dropout(fc1,keep_prob)
# logits = fc_layer(fc1_drop_out,1024,10,"fc2","softmax")
#
# with tf.name_scope('loss'):
#     cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                                   reduction_indices=[1])) # loss
#     tf.summary.scalar('loss', cross_entropy)
#
# with tf.name_scope('train'):
#     train = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)
#
# with tf.name_scope('accuracy'):
#     correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#     tf.summary.scalar('accuracy', accuracy)

########## 测试用 #################


# conv1 layer #

W_conv1 = weight_variable([3,3,1,32])   # patch 3*3,in size 1,out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)     # output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1)                             # output size 14*14*32
tf.summary.histogram("conv1/weights", W_conv1)
tf.summary.histogram("conv1/biases", b_conv1)
tf.summary.histogram("conv1/output", h_pool1)

# conv2 layer #
W_conv2 = weight_variable([3,3,32,64])      # patch 3*3,in size 32,outsize 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) #output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)                         # output size 7*7*64
tf.summary.histogram("conv2/weights", W_conv2)
tf.summary.histogram("conv2/biases", b_conv2)
tf.summary.histogram("conv2/output", h_pool2)

# fc1 layer #
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_fc1_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_fc1_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
tf.summary.histogram("fc1/weights", W_fc1)
tf.summary.histogram("fc1/biases", b_fc1)
tf.summary.histogram("fc1/output", h_fc1_drop)

# fc2 layer #
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
tf.summary.histogram("fc2/weights", W_fc2)
tf.summary.histogram("fc2/biases", b_fc2)
tf.summary.histogram("fc2/output", prediction)


# the error between the prediction and real data
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))   #  loss
tf.summary.scalar('loss',cross_entropy)


with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(ys,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy', accuracy)


with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)


# important step
sess = tf.Session()
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
train_writer = tf.summary.FileWriter("logs/train",sess.graph)
test_writer = tf.summary.FileWriter("logs/test",sess.graph)

sess.run(init)

for i in range(1001):
    if __name__ == '__main__':
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
        if i % 50 == 0:
            precision = compute_accuracy(np.array(mnist.test.images[:1000]),np.array(mnist.test.labels[:1000]))
            train_result = sess.run(merged,feed_dict={xs:np.array(mnist.train.images[:1000]),ys:np.array(mnist.train.labels[:1000]),keep_prob:1})
            test_result = sess.run(merged,feed_dict={xs:np.array(mnist.test.images[:1000]),ys:np.array(mnist.test.labels[:1000]),keep_prob:1})
            train_writer.add_summary(train_result,i)
            test_writer.add_summary(test_result,i)
            print('第%d轮的准确率为：%f'%(i,precision))




