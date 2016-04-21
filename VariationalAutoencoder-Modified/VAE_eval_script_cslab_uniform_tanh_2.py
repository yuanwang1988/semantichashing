import sys
sys.path.append('../')
sys.path.append('./Models/')
sys.path.append('../Utils/')

import cPickle
import gzip

from VAE_eval_script import eval_autoencoder_hashlookup_precision_recall

eval_autoencoder_hashlookup_precision_recall('VAE_uniform_tanh', './results/test_model_uniform_tanh_L12_Noise1/', n_latent=12, prior_noise_level=1, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_uniform_tanh', './results/test_model_uniform_tanh_L12_Noise2/', n_latent=12, prior_noise_level=2, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_uniform_tanh', './results/test_model_uniform_tanh_L12_Noise4/', n_latent=12, prior_noise_level=4, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_uniform_tanh', './results/test_model_uniform_tanh_L12_Noise8/', n_latent=12, prior_noise_level=8, visual_flag=False)

eval_autoencoder_hashlookup_precision_recall('VAE_uniform_tanh', './results/test_model_uniform_tanh_L6_Noise1/', n_latent=6, prior_noise_level=1, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_uniform_tanh', './results/test_model_uniform_tanh_L6_Noise2/', n_latent=6, prior_noise_level=2, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_uniform_tanh', './results/test_model_uniform_tanh_L6_Noise4/', n_latent=6, prior_noise_level=4, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_uniform_tanh', './results/test_model_uniform_tanh_L6_Noise8/', n_latent=6, prior_noise_level=8, visual_flag=False)
