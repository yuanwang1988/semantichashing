from __future__ import absolute_import
from __future__ import print_function
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
from sklearn.manifold import TSNE

#plotting related
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


#import models
from KerasModel import MNIST_autoencoder_784_392_196_98_49_tanh


batch_size = 256
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
print('Initialize Model: MNIST_autoencoder_784_392_196_98_49_tanh')
print('============================')

mnist_autoencoder_784_392_196_98_49_tanh = MNIST_autoencoder_784_392_196_98_49_tanh()


print('============================')
print('Train Model:')
print('============================')

mnist_autoencoder.train(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
       show_accuracy=False, verbose=1, validation_data=[X_test, X_test])

mnist_autoencoder_784_392_196_98_49_tanh.save('./mnist_models/mnist_autoencoder_784_392_196_98_49_tanh')
mnist_autoencoder_784_392_196_98_49_tanh.load('./mnist_models/mnist_autoencoder_784_392_196_98_49_tanh')

print('============================')
print('Evaluate Model:')
print('============================')

score = mnist_autoencoder.evaluate(X_test, X_test)

print('RMSE on validation set: {}'.format(score))