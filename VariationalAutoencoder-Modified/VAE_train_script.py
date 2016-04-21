import sys
sys.path.append('../')
sys.path.append('../Utils')
sys.path.append('./Models')


import numpy as np
import time
import os
from VAE_normal_tanh import VAE as VAE_normal_tanh
from VAE_uniform_tanh import VAE as VAE_uniform_tanh
from VAE_normal import VAE as VAE_normal
import cPickle
import gzip

#custom functions
from hammingHashTable import hammingHashTable, linearLookupTable
from utils import sigmoid, get_cmap


np.random.seed(42)


def train_VAE(VAE_name, VAE_save_folder, continuous = False, \
    hu_encoder=400, hu_decoder=400, n_latent = 20, \
    prior_noise_level = 4, \
    n_epochs = 10, batch_size=256):

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

    #set and create model weight folder (if not already exist)
    path = "{}".format(VAE_save_folder)

    if not os.path.exists(path):
        os.makedirs(path)

    print "instantiating model"
    model =eval('{}(continuous, hu_encoder, hu_decoder, n_latent, x_train, prior_noise_level={}, batch_size=batch_size)'.format(VAE_name, prior_noise_level))

    batch_order = np.arange(int(model.N / model.batch_size))
    epoch = 0
    LB_list = []
    RMSE_train_list = []
    RMSE_valid_list = []

    if os.path.isfile(path + "params.pkl"):
        print "Restarting from earlier saved parameters!"
        model.load_parameters(path)
        LB_list = np.load(path + "LB_list.npy")
        epoch = len(LB_list)

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

        RMSE_train = model.eval_rmse(x_train)
        RMSE_valid = model.eval_rmse(x_valid)

        RMSE_train_list = np.append(RMSE_train_list, RMSE_train)
        RMSE_valid_list = np.append(RMSE_valid_list, RMSE_valid)

        print "Epoch {0} finished. LB: {1}, RMSE train: {2}, RMSE valid: {3}, time: {4}".format(epoch, LB, RMSE_train, RMSE_valid, time.time() - start)
        np.save(path + "LB_list.npy", LB_list)
        np.save(path + "RMSE_train_list", RMSE_train_list)
        np.save(path + "RMSE_valid_list", RMSE_valid_list)

        model.save_parameters(path)

    valid_LB = model.likelihood(x_valid)
    RMSE_valid = model.eval_rmse(x_valid)
    
    print "LB on validation set: {0}".format(valid_LB)
    print "RMSE on validation set: {0}".format(RMSE_valid)


if __name__ == '__main__':
    # train_VAE('VAE_normal_tanh', './test_model_normal_tanh/')
    # train_VAE('VAE_uniform_tanh', './test_model_uniform_tanh/')
    train_VAE('VAE_normal', './results/test_model_normal/', n_latent=20, prior_noise_level=4, n_epochs=10, batch_size=256)