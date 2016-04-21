import sys
sys.path.append('../')
sys.path.append('../Utils/')
sys.path.append('./Models/')

import numpy as np
import time
import os
import cPickle
import gzip

#import models
from VAE_normal_tanh import VAE as VAE_normal_tanh
from VAE_uniform_tanh import VAE as VAE_uniform_tanh
from VAE_normal import VAE as VAE_normal


from sklearn.manifold import TSNE
from sklearn import metrics

#plotting related
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

#custom functions
from hammingHashTable import hammingHashTable, linearLookupTable
from utils import sigmoid, get_cmap

# def eval_autoencoder(autoencoder_name, model_weight_path, n_latent=20, prior_noise_level=4):
# 	eval_autoencoder_RMSE(autoencoder_name, model_weight_path, n_latent=n_latent, prior_noise_level=noise_level)
# 	eval_autoencoder_recon(autoencoder_name, model_weight_path, n_latent=n_latent, prior_noise_level=noise_level)
# 	eval_autoencoder_encode(autoencoder_name, model_weight_path, n_latent=n_latent, prior_noise_level=noise_level)
# 	#eval_autoencoder_hashlookup(autoencoder_name, model_weight_path, n_latent=n_latent, prior_noise_level=noise_level)
# 	eval_autoencoder_hashlookup_precision_recall(autoencoder_name, model_weight_path, n_latent=n_latent, prior_noise_level=noise_level, Limit=250)


def initiate_model(autoencoder_name, model_weight_path, hu_encoder, hu_decoder, n_latent, x_train, prior_noise_level, batch_size=256, continuous=False):
	autoencoder = eval('{}(continuous, hu_encoder, hu_decoder, n_latent, x_train, prior_noise_level={}, batch_size=batch_size)'.format(autoencoder_name, prior_noise_level))
	return autoencoder

def eval_autoencoder_RMSE(autoencoder_name, model_weight_path, n_latent, prior_noise_level):
	print('============================')
	print('Initialize Model: {}_{}'.format(autoencoder_name, prior_noise_level))
	print('============================')

	autoencoder = initiate_model(autoencoder_name, model_weight_path, hu_encoder=400, hu_decoder=400, n_latent=n_latent, x_train=X_train, prior_noise_level=prior_noise_level, batch_size=256)

	autoencoder.load_parameters(model_weight_path)

	print('============================')
	print('Evaluate Model:')
	print('============================')

	score = autoencoder.eval_rmse(X_test)

	print('RMSE on validation set: {}'.format(score))


def eval_autoencoder_recon(autoencoder_name, model_weight_path, n_latent, prior_noise_level, nExamples=10):
	print('============================')
	print('Initialize Model: {}_{}'.format(autoencoder_name, prior_noise_level))
	print('============================')

	autoencoder = initiate_model(autoencoder_name, model_weight_path, hu_encoder=400, hu_decoder=400, n_latent=n_latent, x_train=X_train, prior_noise_level=prior_noise_level, batch_size=256)

	autoencoder.load_parameters(model_weight_path)

	print('============================')
	print('Reconstruction:')
	print('============================')

	z_test = autoencoder.encode(X_test)
	X_test_recon = autoencoder.decode(z_test)

	for i in xrange(nExamples):
		X_test_recon = X_test_recon.reshape((-1,28,28))
		plt.imshow(X_test.reshape((-1,28,28))[i,:,:], cmap=plt.get_cmap("gray"))
		plt.show()

		plt.imshow(X_test_recon[i,:,:], cmap=plt.get_cmap("gray"))
		plt.show()


def eval_autoencoder_encode(autoencoder_name, model_weight_path, n_latent, prior_noise_level):

	print('============================')
	print('Initialize Model: {}_{}'.format(autoencoder_name, prior_noise_level))
	print('============================')

	autoencoder = initiate_model(autoencoder_name, model_weight_path, hu_encoder=400, hu_decoder=400, n_latent=n_latent, x_train=X_train, prior_noise_level=prior_noise_level, batch_size=256)

	autoencoder.load_parameters(model_weight_path)

	print('============================')
	print('Encode:')
	print('============================')

	z_test = autoencoder.encode(X_test)
	z_test = np.tanh(z_test)

	# the histogram of the latent representation
	n, bins, patches = plt.hist(z_test.flatten(), 100, normed=1, facecolor='green', alpha=0.75)

	plt.xlabel('Latent Variable Activation')
	plt.ylabel('Frequency')
	plt.title('Histogram of Activation at Top Layer - Prior Noise = {}'.format(prior_noise_level))
	plt.grid(True)

	plt.show()

	z_mean = np.mean(z_test)
	z_median = np.median(z_test)

	print('Z mean: {}'.format(z_mean))
	print('Z median: {}'.format(z_median))

	# tsne visualization of latent variables
	nExamples = 1000

	cmap = get_cmap(10)
	colour_array = []
	for s in xrange(nExamples):
		colour_array.append(cmap(y_test[s]))


	tsne_model = TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)
	tsne_vec = tsne_model.fit_transform(z_test[0:nExamples,:])

	plt.scatter(tsne_vec[:,0], tsne_vec[:,1], color=colour_array, s=1)
	plt.title('T-SNE of Activation at Top Layer - Prior Noise = {}'.format(prior_noise_level))
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

def eval_autoencoder_hashlookup_precision_recall(autoencoder_name, model_weight_path, n_latent, prior_noise_level, Limit = None, visual_flag = True):

	print('============================')
	print('Initialize Model: {}_{}'.format(autoencoder_name, prior_noise_level))
	print('============================')

	autoencoder = initiate_model(autoencoder_name, model_weight_path, hu_encoder=400, hu_decoder=400, n_latent=n_latent, x_train=X_train, prior_noise_level=prior_noise_level, batch_size=256)

	autoencoder.load_parameters(model_weight_path)

	print('============================')
	print('Encode:')
	print('============================')

	z_test = autoencoder.encode(X_test)

	idx_array = np.zeros((z_test.shape[0], 1), dtype=int)
	for i in xrange(z_test.shape[0]):
		idx_array[i,0] = i



	myTable = linearLookupTable(z_test, X_test)
	myTable2 = linearLookupTable(z_test, idx_array)
	myTable3 = linearLookupTable(z_test, y_test)

	print('============================')
	print('Compute Sample Stats:')
	print('============================')

	print('Frequency of Digits:')
	y_test_freqs= np.bincount(y_test)
	ii = np.nonzero(y_test_freqs)[0]

	print(zip(ii, y_test_freqs[ii]))

	N = z_test.shape[0]
	H = z_test.shape[1]
	if Limit != None:
		N = Limit


	print('============================')
	print('Perform lookup:')
	print('============================')	

	hamming_distance_array = np.arange(H+1)

	n_results_mat = np.zeros((N, H+1))
	precision_mat = np.zeros((N, H+1))
	recall_mat = np.zeros((N, H+1))
	false_pos_rate_mat = np.zeros((N, H+1))

	for i in xrange(N):
		lookup_z = z_test[i,:]
		lookup_y = y_test[i]

		n_results = 0
		true_pos = 0

		for hamming_distance in xrange(H+1):
			resultX, resultZ = myTable.lookup(lookup_z, hamming_distance)
			#resultIdx, _resultZ = myTable2.lookup(lookup_z, hamming_distance)
			#resultY = y_test[resultIdx]
			resultY, _resultZ = myTable3.lookup(lookup_z, hamming_distance)

			n_results = n_results + resultZ.shape[0]
			true_pos = true_pos + np.sum(resultY == lookup_y)

			precision = float(true_pos) / n_results
			recall = float(true_pos) / y_test_freqs[lookup_y]
			false_positive_rate = float(n_results - true_pos)/(z_test.shape[0] - y_test_freqs[lookup_y])

			n_results_mat[i,hamming_distance] = float(n_results)/z_test.shape[0]
			precision_mat[i,hamming_distance] = precision
			recall_mat[i,hamming_distance] = recall
			false_pos_rate_mat[i,hamming_distance] = false_positive_rate

			# print('Example: {}'.format(i))
			# print('Hamm Dist: {}'.format(hamming_distance))
			# print('TP: {}'.format(true_pos))
			# print('n_results: {}'.format(n_results))
			# print('Precision: {}'.format(precision))
			# print('Recall: {}'.format(recall))
			# print('---------------------------------')

		if i%10 == 0:
			print('Finished example {}'.format(i))


	n_results_array = np.mean(n_results_mat, axis=0)
	precision_array = np.mean(precision_mat, axis=0)
	recall_array = np.mean(recall_mat, axis=0)
	false_pos_rate_array = np.mean(false_pos_rate_mat, axis=0)

	if visual_flag:

		#Precision-Recall-NumResults vs. Hamming distance

		n_results_line = plt.plot(hamming_distance_array, n_results_array, label='Num of Results / Total')
		precision_line = plt.plot(hamming_distance_array, precision_array, label='Precision')
		recall_line = plt.plot(hamming_distance_array, recall_array, label='Recall')

		plt.legend()

		plt.xlabel('Hamming Distance')
		plt.ylabel('\%')
		plt.title('Precision-Recall-NumResults vs. Hamming Distance')

		plt.show()

		#Precision recall curve
		plt.plot(recall_array, precision_array)

		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Precision-Recall')

		plt.show()

		#ROC Curve
		plt.plot(false_pos_rate_array, recall_array)

		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic')

		plt.show()


	AUC_score = metrics.auc(false_pos_rate_array, recall_array)

	print('AUC: {}'.format(AUC_score))

	np.savez('{}_L{}_Noise{}_IR_scores'.format(autoencoder_name, n_latent, prior_noise_level), \
		hamming_distance_array=hamming_distance_array, \
		n_results_array=n_results_array, \
		precision_array=precision_array, \
		recall_array=recall_array, \
		false_pos_rate_array=false_pos_rate_array, \
		AUC_score = AUC_score)

	return hamming_distance_array, n_results_array, precision_array, recall_array, false_pos_rate_array, AUC_score

def eval_autoencoder_hashlookup(autoencoder_name, model_weight_path, n_latent, prior_noise_level):

	print('============================')
	print('Initialize Model: {}_{}'.format(autoencoder_name, prior_noise_level))
	print('============================')

	autoencoder = initiate_model(autoencoder_name, model_weight_path, hu_encoder=400, hu_decoder=400, n_latent=n_latent, x_train=X_train, prior_noise_level=prior_noise_level, batch_size=256)

	autoencoder.load_parameters(model_weight_path)

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


if __name__=='__main__':
    print "Loading MNIST data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = gzip.open('mnist.pkl.gz', 'rb')
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = cPickle.load(f)
    f.close()


    eval_autoencoder_encode('VAE_normal_tanh', './results/test_model_normal_tanh_L20_Noise4', n_latent=20, prior_noise_level=4)
    eval_autoencoder_encode('VAE_uniform_tanh', './results/test_model_uniform_tanh_L20_Noise4', n_latent=20, prior_noise_level=4)
