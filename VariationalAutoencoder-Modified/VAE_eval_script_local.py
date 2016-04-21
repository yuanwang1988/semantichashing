import sys
sys.path.append('../')
sys.path.append('./Models/')
sys.path.append('../Utils/')

import cPickle
import gzip

from VAE_eval_script import eval_autoencoder_hashlookup_precision_recall

eval_autoencoder_hashlookup_precision_recall('VAE_normal', './results/test_model_normal_L20_Noise4/', n_latent=20, prior_noise_level=4, Limit=2500)
eval_autoencoder_hashlookup_precision_recall('VAE_normal_tanh', './results/test_model_normal_tanh_L20_Noise4/', n_latent=20, prior_noise_level=4, Limit=2500)
eval_autoencoder_hashlookup_precision_recall('VAE_uniform_tanh', './results/test_model_uniform_tanh_L20_Noise4/', n_latent=20, prior_noise_level=4, Limit=2500)