class Model(object):
	'''
	Goal: The purpose of this wrapper class is to provide scikit-learn like interface to tensorflow

	=============
	Normal Usage:
	=============

	Initialize model: 		model_name = model()
	Load data:				model.load_data(dataset_object)
	Save model weights:		model.save(model_file_path)
	Load model weights:		model.load(model_file_path)
	Train model 			model.train(batch_size, num_iterations)
	Evaluate model 			model.eval()

	'''
	
	def __init__(self):
		'''
		This is a constructor for models built on tensorflow. 

		We specify the attributes that the models must have here but we do not specify the implementation. 
		Each child model needs to specify the implementation of the model by defining the attributes.

		=============
		Attributes:
		=============

		Attributes that need to be set by the child init function:
			- self.sess = ...
			
			- self.train_step = tf function defining a training step
			- self.score = tf function defining how an evaluation is performed

			- self.x = tf placeholder for input features
			- self._y = tf placeholder for target values
			- self.y = output of the model


		Attributes that do not need to be set by child init function:
			- self.data = this is set by calling model.load_data(dataset_object)

		'''


	def save(self, model_file_path):
		'''
		Inputs:
			- model_file_path: string indicating the path to save the model data
		Outputs:
			- None

		Result:
			- model weights are saved to model_file_path
		'''

		saver = tf.train.Saver()
		save_path = saver.save(self.sess, model_file_path)
		print("Model saved in file: %s" % model_file_path)


	def load(self, model_file_path):
		'''
		Inputs:
			- model_file_path: string indicating the path to the model data to be loaded
		Outputs:
			- None

		Result:
			- model weights are loaded from model_file_path
		'''

		saver = tf.train.Saver()
		saver.restore(self.sess, model_file_path)
		print("Model restored from file: %s" % model_file_path)

	def train(self, batch_size, num_iterations):
		'''
		Inputs
			- batch_size: integer indicating size of each batch
			- num_iterations: integer indicating number of training steps

		Outputs:
			- None

		Result:
			- model is trained for num_iterations on the training set
		'''


		for i in range(num_iterations):
			batch = self.data.train.next_batch(batch_size)
			feed = {self.x: batch[0], self.y_: batch[1]}
			self.sess.run(self.train_step, feed_dict=feed)
			if i%50 == 0:
				train_score = self.score.eval(feed_dict={self.x:batch[0], self.y_: batch[1]})
				print "step %d, training score %g"%(i, train_score)

	def eval(self):
		'''
		Inputs:
			- None
		Outputs:
			- test_score - result of the score function evaluated on the test set
		'''

		test_score = self.score.eval(feed_dict={self.x: self.data.test.images, self.y_: self.data.test.labels})
		print "Test set score: %g"%(test_score)

		return test_score

	def load_data(self, dataset_object):
		'''
		Inputs:
			- dataset_object - a tf dataset_object
		Output:
			- None
		Result:
			- Sets the data attribute to the dataset object
		'''

		self.data = dataset_object

	def predictOutput(self, input_features):
		'''
		Inputs:
			- input_features: input features of one test instance
		Outputs:
			- output: predicted output under the model
		'''
		feed_dict = {self.x: input_features}
		output = self.sess.run(self.y, feed_dict)

		#print output

		return output

	def predict(self, input_features):
		output = self.predictOutput(input_features)

		max_index = output.argmax()
		max_val = output[0, max_index]
		
		return (max_index, max_val)


class SimpleSoftMax(Model):
	def __init__(self):
		
		self.sess = tf.InteractiveSession()

		# Create the model
		self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
		self.W = tf.Variable(tf.zeros([784, 10]), name='weights')
		self.b = tf.Variable(tf.zeros([10], name='bias'))

		# Use a name scope to organize nodes in the graph visualizer
		with tf.name_scope('Wx_b'):
			self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

			# Add summary ops to collect data
			_ = tf.histogram_summary('weights', self.W)
			_ = tf.histogram_summary('biases', self.b)
			_ = tf.histogram_summary('y', self.y)

			# Define loss and optimizer
			self.y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
			# More name scopes will clean up the graph representation
		with tf.name_scope('xent'):
			cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
			_ = tf.scalar_summary('cross entropy', cross_entropy)
		with tf.name_scope('train'):
			self.train_step = tf.train.GradientDescentOptimizer(
				0.0001).minimize(cross_entropy)

		with tf.name_scope('test'):
			correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			_ = tf.scalar_summary('accuracy', accuracy)

			self.score = accuracy

		# Merge all the summaries and write them out to /tmp/mnist_logs
		merged = tf.merge_all_summaries()
		writer = tf.train.SummaryWriter('./tmp/mnist_logs', self.sess.graph_def)
		tf.initialize_all_variables().run()


print '===================================='
print 'Testing:'
print '===================================='
from random import randint
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print 'Testing training, saving, loading and evaluating'

SimpleSoftMaxClassifier = SimpleSoftMax()
SimpleSoftMaxClassifier.load_data(mnist)
# SimpleSoftMaxClassifier.train(200, 10000)
# SimpleSoftMaxClassifier.save('./mnist_models/test_model2.cpkt')
SimpleSoftMaxClassifier.load('./mnist_models/test_model2.cpkt')
SimpleSoftMaxClassifier.eval()

print '---------------------'
print 'Testing predict:'
SimpleSoftMaxClassifier.load('./mnist_models/test_model2.cpkt')

for i in xrange(10):

	num = randint(0, mnist.test.images.shape[0])
	img = mnist.test.images[num].reshape(1, 784)

	result_raw = SimpleSoftMaxClassifier.predictOutput(img)
	result = SimpleSoftMaxClassifier.predict(img)

	print 'Results: %d'%(i)
	print 'Most likely class and confidence:'
	print result
	print 'Probability array:'
	print result_raw

	plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
	plt.show()


print '----------------------'