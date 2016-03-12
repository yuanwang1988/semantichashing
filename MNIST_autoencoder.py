from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from matplotlib import pyplot as plt
%matplotlib inline

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.optimizers import RMSprop
from keras.utils import np_utils




batch_size = 64
nb_classes = 10
nb_epoch = 10

# the data, shuffled and split between train and test sets
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

ae2 = Sequential()

encoder2 = containers.Sequential([Dense(input_dim=784, output_dim=392), Dense(input_dim=392, output_dim=196), Dense(input_dim=196, output_dim=98)])
decoder2 = containers.Sequential([Dense(input_dim=98, output_dim=196), Dense(input_dim=196, output_dim=392), Dense(input_dim=392, output_dim=784)])

ae2.add(AutoEncoder(encoder=encoder2, decoder=decoder2, output_reconstruction=True))   #, tie_weights=True))
ae2.compile(loss='mean_squared_error', optimizer=RMSprop())

ae2.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
       show_accuracy=False, verbose=1, validation_data=[X_test, X_test])

y_test2 = ae2.predict(X_test)
y_test2 = y_test2.reshape((-1,28,28))
plt.imshow(y_test2[0,:,:], cmap=plt.get_cmap("gray"))
plt.plot()