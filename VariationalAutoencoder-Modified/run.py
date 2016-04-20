import sys
sys.path.append('../')


import numpy as np
import time
import os
from VAE_uniform_tanh import VAE
import cPickle
import gzip

from sklearn.manifold import TSNE

#plotting related
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

#custom functions
from hammingHashTable import hammingHashTable, linearLookupTable

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

np.random.seed(42)

hu_encoder = 400
hu_decoder = 400
n_latent = 20
continuous = False
n_epochs = 10

if continuous:
    print "Loading Freyface data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = open('freyfaces.pkl', 'rb')
    x = cPickle.load(f)
    f.close()
    x_train = x[:1500]
    x_valid = x[1500:]
else:
    print "Loading MNIST data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = gzip.open('mnist.pkl.gz', 'rb')
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = cPickle.load(f)
    f.close()

path = "./"

print "instantiating model"
model = VAE(continuous, hu_encoder, hu_decoder, n_latent, x_train)


batch_order = np.arange(int(model.N / model.batch_size))
epoch = 0
LB_list = []

if os.path.isfile(path + "params.pkl"):
    print "Restarting from earlier saved parameters!"
    model.load_parameters(path)
    LB_list = np.load(path + "LB_list.npy")
    epoch = len(LB_list)

if __name__ == "__main__":
    print "iterating"
    while epoch < n_epochs:
        epoch += 1
        start = time.time()
        np.random.shuffle(batch_order)
        LB = 0.

        for batch in batch_order:
            batch_LB = model.update(batch, epoch)
            LB += batch_LB

        LB /= len(batch_order)

        LB_list = np.append(LB_list, LB)
        print "Epoch {0} finished. LB: {1}, time: {2}".format(epoch, LB, time.time() - start)
        np.save(path + "LB_list.npy", LB_list)
        model.save_parameters(path)

    valid_LB = model.likelihood(x_valid)
    
    print "LB on validation set: {0}".format(valid_LB)


    x_test = x_test

    z_test = model.encode(x_test)
    z_mu_test = model.encode_mu(x_test)
    z_test_post_activation = sigmoid(z_test)
    #z_sigma_test = model.encode_log_sigma(x_test)
    print 'Validation set X shape - 5 samples: {}'.format(x_test.shape)
    print 'Validation set X shape - 5 samples: {}'.format(z_mu_test.shape)
    #print z_mu_test
    #print z_sigma_test

    x_recon = model.decode(z_mu_test)

    x_mean = (np.mean(x_test, axis=0)).reshape((28,28))

    plt.imshow(x_mean, cmap=plt.get_cmap("gray"))
    plt.show()

    x_test = x_test.reshape((-1,28,28))
    x_recon = x_recon.reshape((-1,28,28))

    for i in xrange(1):
        plt.imshow(x_test[i,:,:], cmap=plt.get_cmap("gray"))
        plt.show()

        plt.imshow(x_recon[i,:,:], cmap=plt.get_cmap("gray"))
        plt.show()

    # print('============================')
    # print('Visualize Latents:')
    # print('============================')

# # the histogram of the data
#     hidden2 = z_test

#     n, bins, patches = plt.hist(hidden2, 100, normed=1, facecolor='green', alpha=0.75)

#     plt.xlabel('Pre-activation')
#     plt.ylabel('Probability')
#     plt.title('Histogram of Z - Prior Noise 4')
#     plt.grid(True)

#     plt.show()

#     hidden2_post_activation = sigmoid(hidden2)

#     # the histogram of the data

#     n, bins, patches = plt.hist(hidden2_post_activation, 100, normed=1, facecolor='green', alpha=0.75)

#     plt.xlabel('Pre-activation')
#     plt.ylabel('Probability')
#     plt.title('Histogram of sigmoid(Z) - Prior Noise 4')
#     plt.grid(True)

#     plt.show()


#     # the histogram of the data
#     hidden2 = z_mu_test

#     n, bins, patches = plt.hist(hidden2, 100, normed=1, facecolor='green', alpha=0.75)

#     plt.xlabel('Pre-activation')
#     plt.ylabel('Probability')
#     plt.title('Histogram of Mu - Prior Noise 4')
#     plt.grid(True)

#     plt.show()

#     hidden2_post_activation = sigmoid(hidden2)

#     # the histogram of the data

#     n, bins, patches = plt.hist(hidden2_post_activation, 100, normed=1, facecolor='green', alpha=0.75)

#     plt.xlabel('Pre-activation')
#     plt.ylabel('Probability')
#     plt.title('Histogram of sigmoid(Mu) - Prior Noise 4')
#     plt.grid(True)

#     plt.show()

#     # the histogram of the data
#     n, bins, patches = plt.hist(z_sigma_test, 100, normed=1, facecolor='green', alpha=0.75)

#     plt.xlabel('Activation')
#     plt.ylabel('Probability')
#     plt.title('Histogram of Log sigmas- Prior Noise 4')
#     plt.grid(True)

#     plt.show()

#     cmap = get_cmap(10)
#     colour_array = []
#     for s in xrange(1000):
#         colour_array.append(cmap(t_test[s]))


#     #tsne based on mean

#     tsne_model = TSNE(n_components=2, random_state=0)
#     np.set_printoptions(suppress=True)
#     tsne_vec = tsne_model.fit_transform(z_mu_test[0:1000, :])

#     plt.scatter(tsne_vec[:,0], tsne_vec[:,1], color=colour_array)
#     plt.show()


#     #tsne based on variance

#     tsne_model = TSNE(n_components=2, random_state=0)
#     np.set_printoptions(suppress=True)
#     tsne_vec = tsne_model.fit_transform(z_sigma_test[0:1000, :])

#     plt.scatter(tsne_vec[:,0], tsne_vec[:,1], color=colour_array)
#     plt.show()


#     #tsne based on combined

#     hidden_complete = np.hstack((z_mu_test, z_sigma_test))

#     tsne_model = TSNE(n_components=2, random_state=0)
#     np.set_printoptions(suppress=True)
#     tsne_vec = tsne_model.fit_transform(hidden2_post_activation[0:1000, :])

#     plt.scatter(tsne_vec[:,0], tsne_vec[:,1], color=colour_array)
#     plt.show()

    print('============================')
    print('Hash Lookup:')
    print('============================')

    y_test = t_test
    X_test = x_test
    z_test = z_test

    y_test_freqs= np.bincount(y_test)
    ii = np.nonzero(y_test_freqs)[0]

    print(zip(ii, y_test_freqs[ii]))

    idx_array = np.zeros((z_test.shape[0], 1))
    for i in xrange(z_test.shape[0]):
        idx_array[i,0] = i

    # myTable = hammingHashTable(z_test, X_test)
    # myTable2 = hammingHashTable(z_test, idx_array)


    myTable = linearLookupTable(z_test, X_test)
    myTable2 = linearLookupTable(z_test, idx_array)


    #choose index of the test example
    i = 652 #652 is one of the few samples that have close by neighbours

    plt.imshow(X_test.reshape((-1,28,28))[i,:,:], cmap=plt.get_cmap("gray"))
    plt.show()

    lookup_z = z_test[i,:]

    print('lookup_z: {}'.format(lookup_z))

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
