from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.append('../')
sys.path.append('../Utils')

import numpy as np
import theano
from utils import sigmoid, get_cmap
from hammingHashTable import hammingHashTable, linearLookupTable

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

#plotting related
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


#import models
from KerasModel import \
MNIST_autoencoder_784_392_196_98_tanh,\
MNIST_autoencoder_784_392_196_98_49_tanh, \
MNIST_autoencoder_784_392_196_98_49_24_tanh, \
MNIST_autoencoder_784_392_196_98_49_20_tanh, \
MNIST_autoencoder_784_392_196_98_49_24_12_tanh, \
MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh, \
Reuters_autoencoder_2000_512_512_128_tanh


def train_autoencoder(autoencoder_name, noise_flag, noise_level):
	print('============================')
	print('Initialize Model: {}_noise={}'.format(autoencoder_name, noise_flag))
	print('============================')


	autoencoder=eval('{}(noise_flag={}, noise_level={})'.format(autoencoder_name, noise_flag, noise_level))


	print('============================')
	print('Train Model:')
	print('============================')

	if noise_flag:
		autoencoder.load('./working_models/{}_{}'.format(autoencoder_name, False))

	hist = autoencoder.train(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
	       show_accuracy=False, verbose=1, validation_data=[X_test, X_test])

	if noise_flag:
		autoencoder.save('./working_models/{}_{}_{}'.format(autoencoder_name, noise_flag, noise_level))
		autoencoder.load('./working_models/{}_{}_{}'.format(autoencoder_name, noise_flag, noise_level))

		np.savez('./working_models/{}_{}_{}_hist'.format(autoencoder_name, noise_flag, noise_level), hist=hist)
	else:
		autoencoder.save('./working_models/{}_{}'.format(autoencoder_name, noise_flag))
		autoencoder.load('./working_models/{}_{}'.format(autoencoder_name, noise_flag))

		np.savez('./working_models/{}_{}_hist'.format(autoencoder_name, noise_flag), hist=hist)


	print('============================')
	print('Evaluate Model:')
	print('============================')

	score = autoencoder.evaluate(X_test, X_test)

	print('RMSE on validation set: {}'.format(score))

	print('################################################################################################################')
	print('################################################################################################################')


if __name__ == "__main__":
	# settings
	batch_size = 256
	nb_classes = 10
	nb_epoch = 1000

	# print('============================')
	# print('Pre-processing data:')
	# print('============================')

	# # the data, shuffled and split between train and test sets
	# (X_train, y_train), (X_test, y_test) = mnist.load_data()
	# X_train = X_train.reshape(-1, 784)
	# X_test = X_test.reshape(-1, 784)
	# X_train = X_train.astype("float32") / 255.0
	# X_test = X_test.astype("float32") / 255.0
	# print(X_train.shape[0], 'train samples')
	# print(X_test.shape[0], 'test samples')

	# models to train

	# train_autoencoder('MNIST_autoencoder_784_392_196_98_tanh')
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_tanh', noise_flag = True)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_tanh')
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_tanh', noise_flag = True)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_20_tanh')
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_20_tanh', noise_flag = True)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_tanh')
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', noise_flag = True)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh')
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', noise_flag = True)

	# train_autoencoder('MNIST_autoencoder_784_392_196_98_tanh', noise_flag = True, noise_level=1)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_tanh', noise_flag = True, noise_level=2)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_tanh', noise_flag = True, noise_level=8)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_tanh', noise_flag = True, noise_level=16)

	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_tanh', noise_flag = True, noise_level=1)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_tanh', noise_flag = True, noise_level=2)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_tanh', noise_flag = True, noise_level=8)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_tanh', noise_flag = True, noise_level=16)

	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_20_tanh', noise_flag = True, noise_level=1)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_20_tanh', noise_flag = True, noise_level=2)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_20_tanh', noise_flag = True, noise_level=8)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_20_tanh', noise_flag = True, noise_level=16)

	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', noise_flag = True, noise_level=1)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', noise_flag = True, noise_level=2)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', noise_flag = True, noise_level=8)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', noise_flag = True, noise_level=16)

	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', noise_flag = True, noise_level=1)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', noise_flag = True, noise_level=2)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', noise_flag = True, noise_level=8)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', noise_flag = True, noise_level=16)

	print('============================')
	print('Pre-processing data:')
	print('============================')

	# settings
	batch_size = 256
	nb_classes = 10
	nb_epoch = 100

	# the data, shuffled and split between train and test sets
	reuters_npz = np.load('./reuters_data/reuters_corpus_tfidf.npz')
	X_train = reuters_npz['X'][0:6000,:]
	X_test = reuters_npz['X'][6000:,:]
	y_train = reuters_npz['y'][0:6000,:]
	y_test = reuters_npz['y'][6000:,:]

	# train_autoencoder('MNIST_autoencoder_784_392_196_98_tanh', noise_flag=False, noise_level=0)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_tanh', noise_flag=True, noise_level=2)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_tanh', noise_flag=True, noise_level=4)

	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_tanh', noise_flag=False, noise_level=0)
	# train_autoencoder('MNIST_autoencoder_784_392_196_98_49_tanh', noise_flag=True, noise_level=2)

	train_autoencoder('Reuters_autoencoder_2000_512_512_128_tanh', noise_flag=False, noise_level=0)
	train_autoencoder('Reuters_autoencoder_2000_512_512_128_tanh', noise_flag=True, noise_level=2)
	train_autoencoder('Reuters_autoencoder_2000_512_512_128_tanh', noise_flag=True, noise_level=4)
