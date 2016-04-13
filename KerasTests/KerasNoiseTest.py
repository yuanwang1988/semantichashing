from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import theano
import math

np.random.seed(1337) # for reproducibility

from matplotlib import pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder, Activation
from keras.layers.noise import GaussianNoise
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_layer_activation(model, test_features, layer_num):
	'''
	Inputs:
		- test features 	- N'xD matrix of test features
		- layer_num 		- integer specifying the layer from which the hidden activation is extracted
	Outputs:
		- hidden_activation	- N'xh matrix of hidden activations
	'''

	get_layer_helper = theano.function([model.layers[0].input], model.layers[layer_num].get_output(train=False), allow_input_downcast=True)
	hidden_activation = get_layer_helper(test_features)

	return hidden_activation

print('=====================')
print('Test Scripts:')
print('=====================')

X_train = np.random.randn(100000, 2)
W_true = np.array([5,1])

y_train = np.round(sigmoid(np.matmul(X_train, W_true)))

print('X_train:')
print(X_train.shape)
print('-----------')

print('W_true')
print(W_true)
print('-----------')

print('y_train')
print(y_train.shape)
print('-----------')


model = Sequential()
model.add(Dense(1, init='uniform', input_dim=2))
# model.add(GaussianNoise(10))
model.add(Activation('sigmoid'))

model.compile(optimizer='RMSprop', loss='binary_crossentropy')

'''
Train the model for 3 epochs, in batches of 16 samples,
on data stored in the Numpy array X_train,
and labels stored in the Numpy array y_train:
'''
model.fit(X_train, y_train, nb_epoch=25, batch_size=128, show_accuracy=True, verbose=1)

print('Evaluate on Training set:')
print(model.evaluate(X_train, y_train, batch_size=128, show_accuracy=True, verbose=1))

print('-----------------')

print('Layer 1: weights')
hidden1 = model.layers[0].get_weights()
print(hidden1)
print('-----------------')

print('Layer 1: Activations')
# hidden1_activations = model.layers[0].get_output(train=False)
# print(dir(hidden1_activations))
hidden1_activations = get_layer_activation(model, X_train, 0)
print('Layer 1 shape: {}'.format(hidden1_activations.shape))
print(hidden1_activations[0:25])
print('-----------------')