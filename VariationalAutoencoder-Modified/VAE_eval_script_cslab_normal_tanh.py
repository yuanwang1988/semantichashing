import sys
sys.path.append('../')
sys.path.append('./Models/')
sys.path.append('../Utils/')

import cPickle
import gzip

from VAE_eval_script import eval_autoencoder_hashlookup_precision_recall

eval_autoencoder_hashlookup_precision_recall('VAE_normal_tanh', './results/test_model_normal_tanh_L49_Noise1/', n_latent=49, prior_noise_level=1, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_normal_tanh', './results/test_model_normal_tanh_L49_Noise2/', n_latent=49, prior_noise_level=2, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_normal_tanh', './results/test_model_normal_tanh_L49_Noise4/', n_latent=49, prior_noise_level=4, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_normal_tanh', './results/test_model_normal_tanh_L49_Noise8/', n_latent=49, prior_noise_level=8, visual_flag=False)

eval_autoencoder_hashlookup_precision_recall('VAE_normal_tanh', './results/test_model_normal_tanh_L20_Noise1/', n_latent=20, prior_noise_level=1, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_normal_tanh', './results/test_model_normal_tanh_L20_Noise2/', n_latent=20, prior_noise_level=2, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_normal_tanh', './results/test_model_normal_tanh_L20_Noise4/', n_latent=20, prior_noise_level=4, visual_flag=False)
eval_autoencoder_hashlookup_precision_recall('VAE_normal_tanh', './results/test_model_normal_tanh_L20_Noise8/', n_latent=20, prior_noise_level=8, visual_flag=False)
