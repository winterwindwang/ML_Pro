import tensorflow as tf
import numpy as np
from CNN import NotMNIST

mnist = NotMNIST.NotMNIST()

# fig = plt.figure(figsize=(8,8))
# for i in range(10):
#     c = 0
#     for (image, label) in zip(mnist.test.images, mnist.test.labels):
#         if np.argmax(label) != i: continue
#         subplot = fig.add_subplot(10,10,i*10+c+1)
#         subplot.set_xticks([])
#         subplot.set_yticks([])
#         subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1,
#                        cmap=plt.cm.gray_r, interpolation="nearest")
#         c += 1
#         if c == 10: break

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,ys:v_ys,keep_prob :1})
    correct_production = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_production,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def biases_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# define placeholder for inputs and outputs
xs = tf.placeholder(tf.float32,[None,784]) / 255.0
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
X_image = tf.reshape(xs,[-1,28,28,1])

# conv2d layer1
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = biases_variable([32])
h_conv1 = tf.nn.relu(conv2d(X_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv2d layer2
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = biases_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fc1 layer
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = biases_variable([1024])
h_flat_fc1 = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_flat_fc1,W_fc1) + b_fc1)
h_fc1_dropout = tf.nn.dropout(h_fc1,keep_prob)

# fc2 layer
W_fc2 = weight_variable([1024,10])
b_fc2 = biases_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_dropout,W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean( -tf.reduce_sum( ys * tf.log(prediction) ,reduction_indices=[1])) # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    if __name__ == '__main__':
        batch_xs,batch_ys = NotMNIST.NotMNIST().train.next_batch(100)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
        if i % 50 == 0:
            print(NotMNIST.NotMNIST().test.labels[:1000])
            print(compute_accuracy(NotMNIST.NotMNIST().test.images[:1000], NotMNIST.NotMNIST().test.labels[:1000]))


