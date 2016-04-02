from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import theano


class KerasModel(object):
	'''
	Goal: The purpose of this wrapper class is to provide scikit-learn like interface to tensorflow

	=============
	Normal Usage:
	=============

	Initialize model: 		model_name = model()
	Save model weights:		model.save(model_file_path)
	Load model weights:		model.load(model_file_path)
	Train model: 			model.train(batch_size, num_iterations)
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





class MNIST_autoencoder(KerasModel):
	def __init__(self):
		ae = Sequential()

		encoder2 = containers.Sequential([Dense(input_dim=784, output_dim=392, activation='relu'), Dense(input_dim=392, output_dim=196, activation='relu'), Dense(input_dim=196, output_dim=98, activation='relu')])     #, GaussianNoise(1), Activation(activation='sigmoid')])
		decoder2 = containers.Sequential([Dense(input_dim=98, output_dim=196, activation='relu'), Dense(input_dim=196, output_dim=392, activation='relu'), Dense(input_dim=392, output_dim=784, activation='softplus')])

		# Dropout.  Not sure if I like it
		#encoder2 = containers.Sequential([Dropout(0.9, input_shape=(784,)), Dense(input_dim=784, output_dim=392, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=392, output_dim=196, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=196, output_dim=98, activation='relu'), Dropout(0.8, input_shape=(98,)), GaussianNoise(1)])
		#decoder2 = containers.Sequential([Dropout(0.8, input_shape=(98,)), Dense(input_dim=98, output_dim=196, activation='relu'), Dropout(0.8, input_shape=(196,)), Dense(input_dim=196, output_dim=392, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=392, output_dim=784)])



		ae.add(AutoEncoder(encoder=encoder2, decoder=decoder2, output_reconstruction=True))   #, tie_weights=True))
		ae.compile(loss='mean_squared_error', optimizer=RMSprop())

		self.model = ae

####################################
#Testing:
####################################


print('################################')
print('Testing:')
print('################################')

np.random.seed(1337) # for reproducibility

from matplotlib import pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils


batch_size = 1000
nb_classes = 10
nb_epoch = 5

print('============================')
print('Pre-processing data:')
print('============================')

# the data, shuffled and split between train and test sets
(X_train, _), (X_test, _) = mnist.load_data()
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

# print(dir(mnist_autoencoder))
# print(dir(mnist_autoencoder.model))
# print(dir(mnist_autoencoder.model.layers))
# print(dir(mnist_autoencoder.model.layers[0]))
# print(dir(mnist_autoencoder.model.layers[0].encoder.layers[0]))
# print(dir(mnist_autoencoder.model.layers[0].encoder.layers[1]))

# print('============================')
# print('Train Model:')
# print('============================')

mnist_autoencoder.train(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
       show_accuracy=False, verbose=1, validation_data=[X_test, X_test])

mnist_autoencoder.save('./mnist_models/keras_autoencoder')
mnist_autoencoder.load('./mnist_models/keras_autoencoder')

print('============================')
print('Evaluate Model:')
print('============================')

score = mnist_autoencoder.evaluate(X_test, X_test)

print('RMSE on validation set: {}'.format(score))

print('============================')
print('Make Predictions:')
print('============================')

y_test2 = mnist_autoencoder.predict(X_test)

for i in xrange(10):
	y_test2 = y_test2.reshape((-1,28,28))
	plt.imshow(X_test.reshape((-1,28,28))[i,:,:], cmap=plt.get_cmap("gray"))
	plt.show()

	plt.imshow(y_test2[i,:,:], cmap=plt.get_cmap("gray"))
	plt.show()


print('============================')
print('Get Hidden Layer:')
print('============================')

print('Layer 1')
hidden1 = mnist_autoencoder.get_autoencoder_layer(X_test, 1)
print(hidden1.shape)
print('------')
print('Layer 2')
hidden2 = mnist_autoencoder.get_autoencoder_layer(X_test, 2)
print(hidden2.shape)
print('------')
print('Layer 3')
hidden3 = mnist_autoencoder.get_autoencoder_layer(X_test, 3)
print(hidden3.shape)
print('------')
print('Layer 4')
hidden4 = mnist_autoencoder.get_autoencoder_layer(X_test, 4)
print(hidden4.shape)
print(hidden4[0])
print('------')