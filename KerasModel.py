from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import theano
from utils import sigmoid, get_cmap
from hammingHashTable import hammingHashTable, linearLookupTable


class KerasModel(object):
	'''
	Goal: The purpose of this wrapper class is to provide scikit-learn like interface to tensorflow

	=============
	Normal Usage:
	=============

	Initialize model: 		model_name = model()
	Save model weights:		model.save(model_file_path)
	Load model weights:		model.load(model_file_path)
	Train model: 			model.train(X_train, y_train, batch_size, num_iterations)
	Evaluate model:			model.evaluate(test_features, test_labels)
	Make prediction: 		model.predict(test_features)
	Get hidden activation:	model.get_layer(test_features, layer_num)
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
			- self.model = ...

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

		self.model.save_weights(model_file_path)

		print('saved model to {}'.format(model_file_path))

	def load(self, model_file_path):
		'''
		Inputs:
			- model_file_path: string indicating the path to the model data to be loaded
		Outputs:
			- None

		Result:
			- model weights are loaded from model_file_path
		'''

		self.model.load_weights(model_file_path)

		print('loaded model from: {}'.format(model_file_path))

	def train(self, train_features, train_targets, batch_size, nb_epoch, show_accuracy=False, verbose=1, validation_data = None):
		'''
		Inputs
			- train_features: 		NxD matrix of training set features
			- train_targets:		Nx1 vector of training set targets
			- batch_size: 			integer indicating size of each batch
			- nb_epoch: 			integer indicating number of training epochs (pass through the entire dataset)
			- validation_data:		[N'xD, N'x1] validation features and vaidation targets

		Outputs:
			- None

		Result:
			- model is trained for nb_epochs on the training set
		'''

		self.model.fit(train_features, train_targets, batch_size=batch_size, nb_epoch=nb_epoch, \
			show_accuracy=show_accuracy, verbose=verbose, validation_data=validation_data)

	def evaluate(self, test_features, test_targets):
		'''
		Inputs:
			- test_set_features - 	NxM array of N examples with M features per example
			- test_set_targets - 	Nxk array of N examples with k possible categories per example
		Outputs:
			- test_score - result of the score function evaluated on the test set
		'''
		score = self.model.evaluate(test_features, test_targets)

		return score

	def predict(self, test_features):
		'''
		Inputs:
			- test_set_features - 	NxM array of N examples with M features per example
		Outputs:
			- test_score - 			predicted target
		'''

		prediction = self.model.predict(test_features)

		return prediction

	def get_autoencoder_layer(self, test_features, layer_num):
		'''
		Inputs:
			- test features 	- N'xD matrix of test features
			- layer_num 		- integer specifying the layer from which the hidden activation is extracted
		Outputs:
			- hidden_activation	- N'xh matrix of hidden activations
		'''

		get_layer_helper = theano.function([self.model.layers[0].encoder.layers[0].input], self.model.layers[0].encoder.layers[layer_num].get_output(train=False), allow_input_downcast=True)
		hidden_activation = get_layer_helper(test_features)

		return hidden_activation

	def encode(self, test_features):
		'''
		Inputs:
			- test features 	- N'xD matrix of test features
		Outputs:
			- z					- N'xh matrix of hidden activations (innermost latent representation)
		'''

		layer_num = len(src_model.model.layers[0].encoder.layers) - 1
		get_layer_helper = theano.function([self.model.layers[0].encoder.layers[0].input], self.model.layers[0].encoder.layers[layer_num].get_output(train=False), allow_input_downcast=True)

		z = get_layer_helper(test_features)

		return z



def autoencoder_transfer_weights(src_model, target_model):
	for k in xrange(len(src_model.model.layers[0].encoder.layers)):
		weights = src_model.model.layers[0].encoder.layers[k].get_weights()
		target_model.model.layers[0].encoder.layers[k].set_weights(weights)

	for k in xrange(len(src_model.model.layers[0].decoder.layers)):
		weights = src_model.model.layers[0].decoder.layers[k].get_weights()
		target_model.model.layers[0].decoder.layers[k].set_weights(weights)



class MNIST_autoencoder(KerasModel):
	def __init__(self):
		'''
		Specify the architecture of the neural network (autoencoder) here.
		'''
		ae = Sequential()

		#encoder without noise
		# encoder2 = containers.Sequential([Dense(input_dim=784, output_dim=392, activation='tanh'), \
		# 	Dense(input_dim=392, output_dim=196, activation='tanh'), \
		# 	Dense(input_dim=196, output_dim=98, activation = 'linear'), Activation(activation='tanh')])
		#encoder with noise
		encoder2 = containers.Sequential([Dense(input_dim=784, output_dim=392, activation='tanh'), \
			Dense(input_dim=392, output_dim=196, activation='tanh'), \
			Dense(input_dim=196, output_dim=98, activation = 'linear'), GaussianNoise(4), Activation(activation='tanh')])
		decoder2 = containers.Sequential([Dense(input_dim=98, output_dim=196, activation='tanh'), \
			Dense(input_dim=196, output_dim=392, activation='tanh'), \
			Dense(input_dim=392, output_dim=784, activation='softplus')])

		# Dropout.  Not sure if I like it
		#encoder2 = containers.Sequential([Dropout(0.9, input_shape=(784,)), Dense(input_dim=784, output_dim=392, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=392, output_dim=196, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=196, output_dim=98, activation='relu'), Dropout(0.8, input_shape=(98,)), GaussianNoise(1)])
		#decoder2 = containers.Sequential([Dropout(0.8, input_shape=(98,)), Dense(input_dim=98, output_dim=196, activation='relu'), Dropout(0.8, input_shape=(196,)), Dense(input_dim=196, output_dim=392, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=392, output_dim=784)])



		ae.add(AutoEncoder(encoder=encoder2, decoder=decoder2, output_reconstruction=True))   #, tie_weights=True))
		ae.compile(loss='mean_squared_error', optimizer=RMSprop())

		self.model = ae

class MNIST_autoencoder_frozen(KerasModel):
	def __init__(self):
		'''
		Specify the architecture of the neural network (autoencoder) here.
		'''

		ae = Sequential()

		#encoder without noise
		# encoder2 = containers.Sequential([Dense(input_dim=784, output_dim=392, activation='tanh'), \
		# 	Dense(input_dim=392, output_dim=196, activation='tanh'), \
		# 	Dense(input_dim=196, output_dim=98, activation = 'linear'), Activation(activation='tanh')])
		#encoder with noise
		# encoder2 = containers.Sequential([Dense(input_dim=784, output_dim=392, activation='tanh'), \
		# 	Dense(input_dim=392, output_dim=196, activation='tanh'), \
		# 	Dense(input_dim=196, output_dim=98, activation = 'linear'), GaussianNoise(4), Activation(activation='tanh')])
		encoder2 = Sequential()
		encoder2.add(Dense(input_dim=784, output_dim=392, activation='tanh', trainable = False))
		encoder2.add(Dense(input_dim=392, output_dim=196, activation='tanh', trainable = False))
		encoder2.add(Dense(input_dim=196, output_dim=98, activation = 'linear', trainable = True))
		encoder2.add(GaussianNoise(4))
		encoder2.add(Activation(activation='tanh'))

		decoder2 = containers.Sequential([Dense(input_dim=98, output_dim=196, activation='tanh', trainable = True), \
			Dense(input_dim=196, output_dim=392, activation='tanh', trainable = False), \
			Dense(input_dim=392, output_dim=784, activation='softplus', trainable = False)])

		# Dropout.  Not sure if I like it
		#encoder2 = containers.Sequential([Dropout(0.9, input_shape=(784,)), Dense(input_dim=784, output_dim=392, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=392, output_dim=196, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=196, output_dim=98, activation='relu'), Dropout(0.8, input_shape=(98,)), GaussianNoise(1)])
		#decoder2 = containers.Sequential([Dropout(0.8, input_shape=(98,)), Dense(input_dim=98, output_dim=196, activation='relu'), Dropout(0.8, input_shape=(196,)), Dense(input_dim=196, output_dim=392, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=392, output_dim=784)])



		ae.add(AutoEncoder(encoder=encoder2, decoder=decoder2, output_reconstruction=True))   #, tie_weights=True))

		
		ae.compile(loss='mean_squared_error', optimizer=RMSprop())

		self.model = ae

	def load(self, model_file_path):
		'''
		Specify the class of the trainable model that will be used to load weights
		'''

		#initiate a trainable model to load the weights
		ae_trainable = MNIST_autoencoder()

		ae_trainable.load(model_file_path)

		#transfer weights from the trainable model:
		for k in xrange(len(ae_trainable.model.layers[0].encoder.layers)):
			weights = ae_trainable.model.layers[0].encoder.layers[k].get_weights()
			self.model.layers[0].encoder.layers[k].set_weights(weights)

		for k in xrange(len(ae_trainable.model.layers[0].decoder.layers)):
			weights = ae_trainable.model.layers[0].decoder.layers[k].get_weights()
			self.model.layers[0].decoder.layers[k].set_weights(weights)


class MNIST_autoencoder_784_392_196_98_49_tanh(KerasModel):
	def __init__(self, noise=False):
		'''
		Specify the architecture of the neural network (autoencoder) here.
		'''
		ae = Sequential()

		#encoder without noise
		# encoder2 = containers.Sequential([Dense(input_dim=784, output_dim=392, activation='tanh'), \
		# 	Dense(input_dim=392, output_dim=196, activation='tanh'), \
		# 	Dense(input_dim=196, output_dim=98, activation = 'linear'), Activation(activation='tanh')])
		#encoder with noise
		encoder = Sequential()
		encoder.add(Dense(input_dim=784, output_dim=392, activation='tanh'))
		encoder.add(Dense(input_dim=392, output_dim=196, activation='tanh'))
		encoder.add(Dense(input_dim=196, output_dim=98, activation = 'tanh'))
		encoder.add(Dense(input_dim=98, output_dim=49, activation = 'linear'))
		encoder.add(GaussianNoise(4))
		encoder.add(Activation(activation='tanh'))
		
		decoder = containers.Sequential([Dense(input_dim=49, output_dim=98, activation='tanh'), \
			Dense(input_dim=98, output_dim=196, activation='tanh'), \
			Dense(input_dim=196, output_dim=392, activation='tanh'), \
			Dense(input_dim=392, output_dim=784, activation='softplus')])

		# Dropout.  Not sure if I like it
		#encoder2 = containers.Sequential([Dropout(0.9, input_shape=(784,)), Dense(input_dim=784, output_dim=392, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=392, output_dim=196, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=196, output_dim=98, activation='relu'), Dropout(0.8, input_shape=(98,)), GaussianNoise(1)])
		#decoder2 = containers.Sequential([Dropout(0.8, input_shape=(98,)), Dense(input_dim=98, output_dim=196, activation='relu'), Dropout(0.8, input_shape=(196,)), Dense(input_dim=196, output_dim=392, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=392, output_dim=784)])



		ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))   #, tie_weights=True))
		ae.compile(loss='mean_squared_error', optimizer=RMSprop())

		self.model = ae



####################################
#Testing:
####################################

if __name__ == "__main__":

	print('################################')
	print('Testing:')
	print('################################')

	np.random.seed(1337) # for reproducibility

	from matplotlib import pyplot as plt

	import keras
	from keras.datasets import mnist
	from keras.models import Sequential
	from keras.layers import containers
	from keras.layers.core import Dense, AutoEncoder, Dropout, Activation
	from keras.layers.noise import GaussianNoise
	from keras.optimizers import RMSprop, Adam
	from keras.utils import np_utils
	from sklearn.manifold import TSNE

	#plotting related
	import matplotlib.pyplot as plt
	import matplotlib.cm as cmx
	import matplotlib.colors as colors


	batch_size = 64
	nb_classes = 10
	nb_epoch = 150

	print('============================')
	print('Pre-processing data:')
	print('============================')

	# the data, shuffled and split between train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.reshape(-1, 784)
	X_test = X_test.reshape(-1, 784)
	X_train = X_train.astype("float32") / 255.0
	X_test = X_test.astype("float32") / 255.0
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	print('============================')
	print('Initialize Model:')
	print('============================')

	mnist_autoencoder = MNIST_autoencoder()

	# # print(dir(mnist_autoencoder))
	# # print(dir(mnist_autoencoder.model))
	# # print(dir(mnist_autoencoder.model.layers))
	# # print(dir(mnist_autoencoder.model.layers[0]))
	# # print(dir(mnist_autoencoder.model.layers[0].encoder.layers[0]))
	# # print(dir(mnist_autoencoder.model.layers[0].encoder.layers[1]))

	print('============================')
	print('Train Model:')
	print('============================')

	mnist_autoencoder.load('./mnist_models/keras_autoencoder')

	# mnist_autoencoder.train(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
	#        show_accuracy=False, verbose=1, validation_data=[X_test, X_test])

	# # mnist_autoencoder.save('./mnist_models/keras_autoencoder_noise4')
	# # mnist_autoencoder.load('./mnist_models/keras_autoencoder_noise4')

	print('============================')
	print('Evaluate Model:')
	print('============================')

	score = mnist_autoencoder.evaluate(X_test, X_test)

	print('RMSE on validation set: {}'.format(score))


	# # print('============================')
	# # print('Make Predictions:')
	# # print('============================')

	# # y_test2 = mnist_autoencoder.predict(X_test)

	# # for i in xrange(10):
	# # 	y_test2 = y_test2.reshape((-1,28,28))
	# # 	plt.imshow(X_test.reshape((-1,28,28))[i,:,:], cmap=plt.get_cmap("gray"))
	# # 	plt.show()

	# # 	plt.imshow(y_test2[i,:,:], cmap=plt.get_cmap("gray"))
	# # 	plt.show()


	# print('============================')
	# print('Get Hidden Layer:')
	# print('============================')

	# print('Layer 1')
	# hidden1 = mnist_autoencoder.get_autoencoder_layer(X_test, 1)
	# print(hidden1.shape)
	# print('------')
	# print('Layer 2')
	# hidden2 = mnist_autoencoder.get_autoencoder_layer(X_test, 2)
	# print(hidden2.shape)
	# print(hidden2[0:25])
	# print('------')

	# # the histogram of the data
	# n, bins, patches = plt.hist(hidden2, 100, normed=1, facecolor='green', alpha=0.75)

	# plt.xlabel('Pre-activation')
	# plt.ylabel('Probability')
	# plt.title('Histogram of Pre-Activation at Top Layer - No Noise')
	# plt.grid(True)

	# plt.show()

	# hidden2_post_activation = sigmoid(hidden2)

	# # the histogram of the data
	# n, bins, patches = plt.hist(hidden2_post_activation, 100, normed=1, facecolor='green', alpha=0.75)

	# plt.xlabel('Activation')
	# plt.ylabel('Probability')
	# plt.title('Histogram of Activation at Top Layer - No Noise')
	# plt.grid(True)

	# plt.show()


	# cmap = get_cmap(10)
	# colour_array = []
	# for s in xrange(1000):
	# 	colour_array.append(cmap(y_test[s]))


	# tsne_model = TSNE(n_components=2, random_state=0)
	# np.set_printoptions(suppress=True)
	# tsne_vec = tsne_model.fit_transform(hidden2_post_activation[0:1000, :])

	# plt.scatter(tsne_vec[:,0], tsne_vec[:,1], color=colour_array)
	# plt.show()

	print('============================')
	print('Initialize Model:')
	print('============================')

	mnist_autoencoder_frozen = MNIST_autoencoder_frozen()

	# print('============================')
	# print('Train Model:')
	# print('============================')

	# mnist_autoencoder_frozen.load('./mnist_models/keras_autoencoder')

	# mnist_autoencoder_frozen.train(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
	#        show_accuracy=False, verbose=1, validation_data=[X_test, X_test])

	# mnist_autoencoder_frozen.save('./mnist_models/keras_autoencoder_noise4_partial_freeze')
	mnist_autoencoder_frozen.load('./mnist_models/keras_autoencoder_noise4_partial_freeze')

	print('============================')
	print('Evaluate Model:')
	print('============================')

	score = mnist_autoencoder_frozen.evaluate(X_test, X_test)

	print('RMSE on validation set: {}'.format(score))


	# print('============================')
	# print('Make Predictions:')
	# print('============================')

	# y_test2 = mnist_autoencoder_frozen.predict(X_test)

	# for i in xrange(10):
	# 	y_test2 = y_test2.reshape((-1,28,28))
	# 	plt.imshow(X_test.reshape((-1,28,28))[i,:,:], cmap=plt.get_cmap("gray"))
	# 	plt.show()

	# 	plt.imshow(y_test2[i,:,:], cmap=plt.get_cmap("gray"))
	# 	plt.show()


	print('============================')
	print('Get Hidden Layer:')
	print('============================')

	print('Layer 1')
	hidden1 = mnist_autoencoder_frozen.get_autoencoder_layer(X_test, 1)
	print(hidden1.shape)
	print('------')
	print('Layer 2')
	hidden2 = mnist_autoencoder_frozen.get_autoencoder_layer(X_test, 2)
	print(hidden2.shape)
	print(hidden2[0:25])
	print('------')

	hidden2_post_activation = sigmoid(hidden2)

	# # the histogram of the data
	# n, bins, patches = plt.hist(hidden2, 100, normed=1, facecolor='green', alpha=0.75)

	# plt.xlabel('Pre-activation')
	# plt.ylabel('Probability')
	# plt.title('Histogram of Pre-Activation at Top Layer - Gaussian Noise (STD = 4)')
	# plt.grid(True)

	# plt.show()

	# # the histogram of the data
	# n, bins, patches = plt.hist(hidden2_post_activation, 100, normed=1, facecolor='green', alpha=0.75)

	# plt.xlabel('Activation')
	# plt.ylabel('Probability')
	# plt.title('Histogram of Activation at Top Layer - Gaussian Noise (STD=4)')
	# plt.grid(True)

	# plt.show()

	# cmap = get_cmap(10)
	# colour_array = []
	# for s in xrange(1000):
	# 	colour_array.append(cmap(y_test[s]))


	# tsne_model = TSNE(n_components=2, random_state=0)
	# np.set_printoptions(suppress=True)
	# tsne_vec = tsne_model.fit_transform(hidden2_post_activation[0:1000, :])

	# plt.scatter(tsne_vec[:,0], tsne_vec[:,1], color=colour_array)
	# plt.show()


	# cmap = get_cmap(10)
	# colour_array = []
	# idx_array = np.zeros((10,1))
	# for s in xrange(10):
	# 	idx_array[s,0] = s+1
	# 	colour_array.append(cmap(s+1))

	# plt.scatter(idx_array[:,0], idx_array[:,0], color=colour_array)
	# plt.show()

	print('============================')
	print('Hash Lookup:')
	print('============================')

	y_test_freqs= np.bincount(y_test)
	ii = np.nonzero(y_test_freqs)[0]

	print(zip(ii, y_test_freqs[ii]))

	idx_array = np.zeros((hidden2_post_activation.shape[0], 1))
	for i in xrange(hidden2_post_activation.shape[0]):
		idx_array[i,0] = i

	# myTable = hammingHashTable(hidden2_post_activation, X_test)
	# myTable2 = hammingHashTable(hidden2_post_activation, idx_array)


	myTable = linearLookupTable(hidden2_post_activation, X_test)
	myTable2 = linearLookupTable(hidden2_post_activation, idx_array)


	#choose index of the test example
	i = 652 #652 is one of the few samples that have close by neighbours

	plt.imshow(X_test.reshape((-1,28,28))[i,:,:], cmap=plt.get_cmap("gray"))
	plt.show()

	lookup_z = hidden2_post_activation[i,:]

	print('hamming distance of 1')
	resultX, resultZ = myTable.lookup(lookup_z, 1)
	resultIdx, _resultZ = myTable2.lookup(lookup_z, 1)

	print('Shape of results: {}'.format(resultX.shape))
	for j in xrange(resultX.shape[0]):
		print('Latent Z: {}'.format(resultZ[j,:]))
		print('Index: {}'.format(resultIdx[j]))
		fig = plt.figure()
		plt.imshow(resultX[j,:].reshape((28,28)), cmap=plt.get_cmap("gray"))
		plt.draw()
		plt.pause(1) # <-------
		raw_input("<Hit Enter To Close>")
		plt.close(fig)
		print('-------')

	print('hamming distance of 2')
	resultX, resultZ = myTable.lookup(lookup_z, 2)
	resultIdx, _resultZ = myTable2.lookup(lookup_z, 2)

	print('Shape of results: {}'.format(resultX.shape))
	for j in xrange(resultX.shape[0]):
		print('Latent Z: {}'.format(resultZ[j,:]))
		print('Index: {}'.format(resultIdx[j]))
		fig = plt.figure()
		plt.imshow(resultX[j,:].reshape((28,28)), cmap=plt.get_cmap("gray"))
		plt.draw()
		plt.pause(1) # <-------
		raw_input("<Hit Enter To Close>")
		plt.close(fig)
		print('-------')

	print('hamming distance of 3')
	resultX, resultZ = myTable.lookup(lookup_z, 3)
	resultIdx, _resultZ = myTable2.lookup(lookup_z, 3)

	print('Shape of results: {}'.format(resultX.shape))
	for j in xrange(resultX.shape[0]):
		print('Latent Z: {}'.format(resultZ[j,:]))
		print('Index: {}'.format(resultIdx[j]))
		fig = plt.figure()
		plt.imshow(resultX[j,:].reshape((28,28)), cmap=plt.get_cmap("gray"))
		plt.draw()
		plt.pause(1) # <-------
		raw_input("<Hit Enter To Close>")
		plt.close(fig)
		print('-------')
