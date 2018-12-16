import tensorflow as tf
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utilities import *

input_file = r''
# 求二阶差分
def second_order_derivative(signal):
    return np.array([signal[i+1] - 2*signal[i] - signal[i-1] for i in range(len(signal)-2) if i>=1])

def get_feature(input_file, isStego=1):
    files = os.listdir(input_file)
    files = [input_file + '\\' + f for f in files if f.endswith('.wav')]
    mfccs = []
    labels = []
    for i in range(1000):
        y, sr = librosa.load(files[i], sr=None, duration=1)  # origin smaplerate:16k channels:1  duration:3.968
        # y = second_order_derivative(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)  # (24,32)
        mfcc = np.mean(mfcc, axis=0).transpose()
        mfccs.append(mfcc)  # feature shape (32,)
        labels.append(isStego)
    return np.array(mfccs), np.array(labels)


X_train , label_train, list_ch_train  =read_data(data_file='./data', split='train')  # train
X_test, label_test, list_ch_test = read_data(data_file='./data', split='train')  # test
assert list_ch_train == list_ch_test , 'Mistmatch in channels'

# Normalize
X_train, X_test = standardize(X_train,X_test)
X_tr, X_vld, label_tr, label_vld = train_test_split(X_train, label_train, stratify=label_train,random_state=123)
# One-hot-encoding
y_tr = one_hot(label_tr)
y_vld = one_hot(label_vld)
y_test = one_hot(label_test)

# Hyperparaments
batch_size=600   # batch size
seq_len = 128    # Number of steps
learning_rate = 0.0001
epochs = 1000

n_channels = 9
n_classes = 6

# placeholder
graph = tf.Graph()
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32,[None,seq_len, n_channels], name='inputs')
    labels_ = tf.placeholder(tf.float32,[None,n_classes], name='labels')
    keep_prob_ = tf.placeholder(tf.float32, name='keep')
    learning_rate_ = tf.placeholder(tf.float32, name='learning rate')

# convolutional layers
with graph.as_default():
    # (batch, 128, 9) -->(batch, 64, 18)
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=18, kernel_size=2, strides=1, padding='SAME',activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='SAME')

    # (batch, 64,18)--->(batch, 32,36)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1, padding='SAME',activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2,padding='SAME')

    # (batch, 32, 36) --->(batch, 18, 72)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1, padding='SAME',
                             activation=tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2,padding='SAME')

 # (batch, 16, 72) --->(batch, 8, 144)
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1, padding='SAME',
                             activation=tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2,padding='SAME')

# flatten and pass to classifier
with graph.as_default():
    # flatten and dropout
    flat = np.reshape(max_pool_4,(-1,8*144))
    flat = tf.nn.dropout(flat,keep_prob=keep_prob_)

    # predictiona
    logits = tf.layers.dense(flat, n_classes)

    # cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

    # accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


def get_batches(x):
    return x

# train the network
validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1

    # Loop over epochs
    for e in range(epochs):

        # Loop over batches
        for x, y in get_batches(X_tr, y_tr, batch_size):

            # Feed dictionary
            feed = {inputs_: x, labels_: y, keep_prob_: 0.5, learning_rate_: learning_rate}

            # Loss
            loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict=feed)
            train_acc.append(acc)
            train_loss.append(loss)

            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))

            # Compute validation loss at every 10 iterations
            if (iteration % 10 == 0):
                val_acc_ = []
                val_loss_ = []

                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_: x_v, labels_: y_v, keep_prob_: 1.0}

                    # Loss
                    loss_v, acc_v = sess.run([cost, accuracy], feed_dict=feed)
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)

                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))

            # Iterate
            iteration += 1

    saver.save(sess, "checkpoints-cnn/har.ckpt")

# Plot training and test loss
t = np.arange(iteration-1)

plt.figure(figsize = (6,6))
plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# Plot Accuracies
plt.figure(figsize = (6,6))

plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
plt.xlabel("iteration")
plt.ylabel("Accuray")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()