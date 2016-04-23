from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.append('../')
sys.path.append('../Utils')

import numpy as np
import keras
from keras.datasets import mnist

from sklearn import lda, decomposition


class ScikitModel(object):
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

	def train(self, X_train, y_train):
		self.model.fit(X_train, y_train)
		self.means = np.mean(self.model.transform(X_train), axis=0)


	def encode(self, test_features):
		'''
		Inputs:
			- test features 	- N'xD matrix of test features
		Outputs:
			- z					- N'xh matrix of hidden activations (innermost latent representation)
		'''
		z = self.model.transform(test_features) - self.means

		return z


class LDA_model(ScikitModel):
	def __init__(self, n_components):
		self.model = lda.LDA(n_components=n_components)


class PCA_model(ScikitModel):
	def __init__(self, n_components):
		self.model = decomposition.PCA(n_components=n_components)


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

	lda_model_20 = LDA_model(n_components=20)

	lda_model_20.train(X_train, y_train)
	
	print(lda_model_20.encode(X_test).shape)
	print(lda_model_20.model.score(X_test, y_test))