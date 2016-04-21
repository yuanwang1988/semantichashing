import tensorflow as tf 
import numpy as np 

#load mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#start interactive session
sess = tf.InteractiveSession()

#=================
#helper functions:
#=================

#functions for setting weights and bias parameters:
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#functions for building convolution layers and max pool layers:
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#===================
#Script:
#===================

#set input and output nodes -> placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)


#set paramters
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

sess.run(tf.initialize_all_variables())

#set prediction function

#input layer
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#drop out layer
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#output layer
y = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

#set cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#set training/optimization:
train_step = tf.train.GradientDescentOptimizer(0.00005).minimize(cross_entropy)

#compute accuracy
#compute correctness:
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#compute accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#train:
for i in range(20000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
  #report every 100 iterations:
  if i%50 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob:1})
    print("step %d, training accuracy %g"%(i, train_accuracy))



#report accuracy
print("Test set accuracy: {}".format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1})))




