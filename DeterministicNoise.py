
# coding: utf-8


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')

sess = tf.InteractiveSession()

#########################################################
#  Very simple model with 1 data point and no noise.  The purpose of this 
# is to practice with basic syntax and to see if I can get gradients
#########################################################

# Set up the placeholders for x and y
x = tf.placeholder(np.float32, shape=(2,1), name='x')
y = tf.placeholder(np.float32, shape=(1,1), name='y')

# Set up trainable variables
W = tf.Variable( np.array([[-5,3]], dtype=np.float32).reshape(2,1))
b = tf.Variable( np.array([-2], dtype=np.float32))

# Set up a simple matrix multiply operation
y_ = tf.add(tf.matmul(W,x, transpose_a=True, name="y_"),b)

# Set up a gradient operation
grad = tf.gradients(ys=y_, xs=x, name="grad")

# Initialize all variables
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)


# Make an actual data point
x_data = np.array([1,1], dtype=np.float32).reshape(2,1)
y_data = np.array(7, dtype=np.float32).reshape(1,1)

fd = {x:x_data, y:y_data}


# See if it's working
print sess.run(y_, feed_dict=fd)


# See if the gradient is working.  It should just be the weights
print sess.run(grad, feed_dict=fd)

#########################################################
# # Now let's add noise and see if we can still get the right gradient.  Should noise be a placeholder, variable, or constant?
#########################################################

noise = tf.placeholder(dtype=np.float32, shape=y_.get_shape(), name="noise")

y_noise = y_ + noise
grad_noise = tf.gradients(ys=y_noise, xs=x, name="grad_noise")


fd_noise = {x:x_data, y:y_data, noise:np.random.randn(1,1).astype(np.float32)}

print sess.run(y_noise, feed_dict=fd_noise)
print sess.run(grad_noise, feed_dict=fd_noise)

# # Success!


#########################################################
# # Now let's try to learn a linear regression model with deterministic noise.  1000 data points this time, and we'll minimize MSE
#########################################################


N = 1000  # number of data points

W_actual = np.array([4, -3], dtype=np.float32).reshape(2,1)
b_actual = np.array(2, dtype=np.float32).reshape(1,1)
x_data = np.random.rand(N,2).astype(np.float32)
y_data = x_data.dot(W_actual) + b_actual
noise_data = 10 * np.random.randn(N).reshape(-1,1).astype(np.float32)


# Setup placeholders
x = tf.placeholder(dtype=np.float32, shape=(None, 2), name='x')
y = tf.placeholder(dtype=np.float32, shape=(None, 1), name='y')
noise = tf.placeholder(dtype=np.float32, shape=(None,1), name='noise')

# Setup trainable variables
W = tf.Variable( np.random.rand(2,1).astype(np.float32), name='W')
b = tf.Variable(np.random.rand(1,1).astype(np.float32), name='b')

# Setup operations
y_ = tf.add( tf.matmul(x,W)     , b  , name='y_')
loss = tf.reduce_mean(tf.square(y - y_), name = 'mse')
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss, name='train')
grad = tf.gradients(y_, x, name='grad')

# Initialize all variables
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)


iters = 100

feed_dict={x:x_data, y:y_data, noise:noise_data}


for i in np.arange(iters):
     sess.run(train, feed_dict)


print sess.run(W)
print sess.run(b)


sess.run(grad, feed_dict)

# Success!



