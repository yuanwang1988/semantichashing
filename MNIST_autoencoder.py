
# coding: utf-8

# In[81]:

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils



# In[62]:

nb_classes = 10

# the data, shuffled and split between train and test sets
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[96]:

ae2 = Sequential()

encoder2 = containers.Sequential([Dense(input_dim=784, output_dim=392, activation='relu'), Dense(input_dim=392, output_dim=196, activation='relu'), Dense(input_dim=196, output_dim=98, activation='relu')])     #, GaussianNoise(1), Activation(activation='sigmoid')])
decoder2 = containers.Sequential([Dense(input_dim=98, output_dim=196, activation='relu'), Dense(input_dim=196, output_dim=392, activation='relu'), Dense(input_dim=392, output_dim=784, activation='softplus')])

# Dropout.  Not sure if I like it
#encoder2 = containers.Sequential([Dropout(0.9, input_shape=(784,)), Dense(input_dim=784, output_dim=392, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=392, output_dim=196, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=196, output_dim=98, activation='relu'), Dropout(0.8, input_shape=(98,)), GaussianNoise(1)])
#decoder2 = containers.Sequential([Dropout(0.8, input_shape=(98,)), Dense(input_dim=98, output_dim=196, activation='relu'), Dropout(0.8, input_shape=(196,)), Dense(input_dim=196, output_dim=392, activation='relu'), Dropout(0.8, input_shape=(392,)), Dense(input_dim=392, output_dim=784)])



ae2.add(AutoEncoder(encoder=encoder2, decoder=decoder2, output_reconstruction=True))   #, tie_weights=True))
ae2.compile(loss='mean_squared_error', optimizer=RMSprop())


# In[97]:

nb_epoch = 10
batch_size = 64


ae2.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
       show_accuracy=False, verbose=1, validation_data=[X_test, X_test])



# In[99]:

idx = 49

y_test2 = ae2.predict(X_test);
y_test2 = y_test2.reshape((-1,28,28));

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(X_test[idx].reshape(28,28), cmap=plt.get_cmap("gray"))
ax1.set_title("Original")
ax2.imshow(y_test2[idx,:,:], cmap=plt.get_cmap("gray"))
ax2.set_title("Reconstruction")
plt.show()


# In[ ]:




# In[ ]:



