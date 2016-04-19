from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import theano
from utils import sigmoid, get_cmap
from hammingHashTable import hammingHashTable, linearLookupTable

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

#import models
from KerasModel import \
MNIST_autoencoder_784_392_196_98_tanh,\
MNIST_autoencoder_784_392_196_98_49_tanh, \
MNIST_autoencoder_784_392_196_98_49_24_tanh, \
MNIST_autoencoder_784_392_196_98_49_20_tanh, \
MNIST_autoencoder_784_392_196_98_49_24_12_tanh, \
MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh

np.random.seed(1337) # for reproducibility

def eval_autoencoder(autoencoder_name, model_weight_path, noise_flag=False, noise_level=4):
	eval_autoencoder_RMSE(autoencoder_name, model_weight_path, noise_flag=False, noise_level=4)
	eval_autoencoder_recon(autoencoder_name, model_weight_path, noise_flag=False, noise_level=4)
	eval_autoencoder_encode(autoencoder_name, model_weight_path, noise_flag=False, noise_level=4)
	eval_autoencoder_hashlookup(autoencoder_name, model_weight_path, noise_flag=False, noise_level=4)


def eval_autoencoder_RMSE(autoencoder_name, model_weight_path, noise_flag=False, noise_level=4):
	print('============================')
	print('Initialize Model: {}_{}'.format(autoencoder_name, noise_flag))
	print('============================')

	autoencoder = eval('{}(noise_flag={})'.format(autoencoder_name, noise_flag, noise_level))

	autoencoder.load(model_weight_path)

	print('============================')
	print('Evaluate Model:')
	print('============================')

	score = autoencoder.evaluate(X_test, X_test)

	print('RMSE on validation set: {}'.format(score))


def eval_autoencoder_recon(autoencoder_name, model_weight_path, noise_flag=False, noise_level=4):
	print('============================')
	print('Initialize Model: {}_{}'.format(autoencoder_name, noise_flag))
	print('============================')

	autoencoder = eval('{}(noise_flag={})'.format(autoencoder_name, noise_flag, noise_level))

	autoencoder.load(model_weight_path)

	print('============================')
	print('Reconstruction:')
	print('============================')

	x_test_recon = autoencoder.predict(X_test)


	for i in xrange(10):
		x_test_recon = x_test_recon.reshape((-1,28,28))
		plt.imshow(X_test.reshape((-1,28,28))[i,:,:], cmap=plt.get_cmap("gray"))
		plt.show()

		plt.imshow(x_test_recon[i,:,:], cmap=plt.get_cmap("gray"))
		plt.show()


def eval_autoencoder_encode(autoencoder_name, model_weight_path, noise_flag=False, noise_level=4):

	print('============================')
	print('Initialize Model: {}_{}'.format(autoencoder_name, noise_flag))
	print('============================')

	autoencoder = eval('{}(noise_flag={})'.format(autoencoder_name, noise_flag, noise_level))

	autoencoder.load(model_weight_path)

	print('============================')
	print('Encode:')
	print('============================')

	z_test = autoencoder.encode(X_test)

	# the histogram of the latent representation
	n, bins, patches = plt.hist(z_test, 100, normed=1, facecolor='green', alpha=0.75)

	plt.xlabel('Latent Variable Activation')
	plt.ylabel('Frequency')
	plt.title('Histogram of Activation at Top Layer - Gaussian Noise = {}'.format(noise_flag))
	plt.grid(True)

	plt.show()

	# tsne visualization of latent variables
	cmap = get_cmap(10)
	colour_array = []
	for s in xrange(1000):
		colour_array.append(cmap(y_test[s]))


	tsne_model = TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)
	tsne_vec = tsne_model.fit_transform(z_test[0:1000,:])

	plt.scatter(tsne_vec[:,0], tsne_vec[:,1], color=colour_array)
	plt.title('T-SNE of Activation at Top Layer - Gaussian Noise = {}'.format(noise_flag))
	plt.show()


	cmap = get_cmap(10)
	colour_array = []
	idx_array = np.zeros((10,1))
	for s in xrange(10):
		idx_array[s,0] = s+1
		colour_array.append(cmap(s+1))

	plt.scatter(idx_array[:,0], idx_array[:,0], color=colour_array)
	plt.title('T-SNE of Activation at Top Layer - Colour Legend')
	plt.show()


def eval_autoencoder_hashlookup(autoencoder_name, model_weight_path, noise_flag=False, noise_level=4):

	print('============================')
	print('Initialize Model: {}_{}'.format(autoencoder_name, noise_flag))
	print('============================')

	autoencoder = eval('{}(noise_flag={})'.format(autoencoder_name, noise_flag, noise_level))

	autoencoder.load(model_weight_path)

	print('============================')
	print('Hash Lookup:')
	print('============================')

	z_test = autoencoder.encode(X_test)

	print('Frequency of Digits:')
	y_test_freqs= np.bincount(y_test)
	ii = np.nonzero(y_test_freqs)[0]

	print(zip(ii, y_test_freqs[ii]))

	idx_array = np.zeros((z_test.shape[0], 1))
	for i in xrange(z_test.shape[0]):
		idx_array[i,0] = i


	myTable = linearLookupTable(z_test, X_test)
	myTable2 = linearLookupTable(z_test, idx_array)


	#choose index of the test example
	i = 652 #652 is one of the few samples that have close by neighbours

	plt.imshow(X_test.reshape((-1,28,28))[i,:,:], cmap=plt.get_cmap("gray"))
	plt.show()

	lookup_z = z_test[i,:]

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




if __name__ == '__main__':
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


	eval_autoencoder('MNIST_autoencoder_784_392_196_98_49_tanh', './mnist_models/mnist_autoencoder_784_392_196_98_49_tanh_False')
	eval_autoencoder('MNIST_autoencoder_784_392_196_98_49_tanh', './mnist_models/mnist_autoencoder_784_392_196_98_49_tanh_True')


