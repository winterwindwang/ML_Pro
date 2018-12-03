import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

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

def conv_layer(inputs, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        w = weight_variable([3,3,channels_in,channels_out])
        b = bias_variable([channels_out])
        conv = tf.nn.relu(conv2d(inputs, w ) + b)
        return max_pool_2x2(conv)

def fc_layer(inputs, channel_in, channel_out, name="fc",fun=None):
    with tf.name_scope(name):
        w = weight_variable([channel_in,channel_out])
        b = bias_variable([channel_out])
        if fun is not None:
            return tf.nn.relu(tf.matmul(inputs, w) + b)
        else:
            return tf.nn.softmax(tf.matmul(inputs, w) + b)

conv1=conv_layer(x_image,1,32,'conv1')
conv2=conv_layer(conv1,32,64,'conv2')

flattened = tf.reshape(conv2,[-1,7*7*64])
fc1 = fc_layer(flattened,7*7*64,1024,"fc1")
fc1_drop_out = tf.nn.dropout(fc1,keep_prob)
logits = fc_layer(fc1_drop_out,1024,10,"fc2","softmax")

W = weight_variable([1024,10])
b = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(fc1,W) + b)

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1])) # loss
    tf.summary.scalar('loss', cross_entropy)

with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy', accuracy)

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
