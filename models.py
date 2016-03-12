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
	Evaluate model 			model.eval(test_features, test_labels)

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

	def eval(self, test_set_features, test_set_targets):
		'''
		Inputs:
			- test_set_features - NxM array of N examples with M features per example
			- test_set_targets - Nxk array of N examples with k possible categories per example
		Outputs:
			- test_score - result of the score function evaluated on the test set
		'''

		test_score = self.score.eval(feed_dict={self.x: test_set_features, self.y_: test_set_targets})
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

	def load_train_save(self, in_model_name, out_model_name, batch_size, num_iterations):
		self.load(in_model_name)
		self.train(batch_size, num_iterations)
		self.save(out_model_name)


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


class AutoEncoder(Model):

	def __init__(self, layer_sizes):
		self.sess = tf.InteractiveSession()
		self.x = tf.placeholder(tf.float32, [None, 784])
		self.y_ = tf.placeholder(tf.float32, [None, 10])


		# Build the encoding layers
		next_layer_input = self.x

		encoding_matrices = []
		for dim in layer_sizes:
			input_dim = int(next_layer_input.get_shape()[1])

			# Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
			#W = tf.Variable(tf.random_uniform([input_dim, dim], 1, 1.0))

			#tempoary hack - try to initialize W = identify
			diagonal = tf.ones([input_dim], tf.float32)
			W = tf.Variable(tf.add(tf.diag(diagonal), tf.random_uniform([input_dim, dim], -0.1, 0.1)))

			# Initialize b to zero
			b = tf.Variable(tf.zeros([dim]))

			# We are going to use tied-weights so store the W matrix for later reference.
			encoding_matrices.append(W)

			output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)

			# the input into the next layer is the output of this layer
			next_layer_input = output

		# The fully encoded x value is now stored in the next_layer_input
		encoded_x = next_layer_input

		# build the reconstruction layers by reversing the reductions
		layer_sizes.reverse()
		encoding_matrices.reverse()


		for i, dim in enumerate(layer_sizes[1:] + [ int(self.x.get_shape()[1])]) :
			# we are using tied weights, so just lookup the encoding matrix for this step and transpose it
			W = tf.transpose(encoding_matrices[i])
			b = tf.Variable(tf.zeros([dim]))
			output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)
			next_layer_input = output

		# the fully encoded and reconstructed value of x is here:
		reconstructed_x = next_layer_input

		self.y = reconstructed_x
		self.score = tf.sqrt(tf.reduce_mean(tf.square(self.x-reconstructed_x)))

		init = tf.initialize_all_variables()
		self.sess.run(init)


		self.train_step = tf.train.AdagradOptimizer(0.001).minimize(self.score)

		init = tf.initialize_all_variables()
		self.sess.run(init)

class VariationalAutoencoder(Model):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                 biases['out_mean']))
        return x_reconstr_mean
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.score = self.cost
        self.train_step = self.optimizer
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})




print '===================================='
print 'Testing:'
print '===================================='
import math
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
print mnist.test.images.shape
print mnist.test.labels.shape

SimpleSoftMaxClassifier.eval(mnist.test.images, mnist.test.labels)

print '---------------------'
print 'Testing predict:'
SimpleSoftMaxClassifier.load('./mnist_models/test_model2.cpkt')

for i in xrange(0):

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

# print 'Test AutoEncoder'

#AutoEncoder = AutoEncoder([784, 392, 196, 98, 49, 7])
# AutoEncoder = AutoEncoder([784])
# AutoEncoder.load_data(mnist)
# AutoEncoder.load('./mnist_models/autoEncoderTestModel3.cpkt')
# AutoEncoder.train(1000, 1000)
# AutoEncoder.save('./mnist_models/autoEncoderTestModel3.cpkt')
# AutoEncoder.load('./mnist_models/autoEncoderTestModel3.cpkt')

# AutoEncoder.eval(mnist.test.images, mnist.test.labels)

# print '----------------------'

# print 'Testing predict:'
# AutoEncoder.load('./mnist_models/autoEncoderTestModel3.cpkt')

# for i in xrange(5):

# 	num = randint(0, mnist.test.images.shape[0])
# 	img = mnist.test.images[num].reshape(1, 784)

# 	result_raw = AutoEncoder.predictOutput(img)

# 	plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
# 	plt.show()

# 	plt.imshow(result_raw.reshape(28, 28), cmap=plt.cm.binary)
# 	plt.show()


# print '----------------------'