import sys
sys.path.append('../')

from VAE_train_script import train_VAE

train_VAE('VAE_normal', '../results/test_model_normal_L12_Noise1/', n_latent=12, prior_noise_level=1, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', '../results/test_model_normal_L12_Noise2/', n_latent=12, prior_noise_level=2, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', '../results/test_model_normal_L12_Noise4/', n_latent=12, prior_noise_level=4, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', '../results/test_model_normal_L12_Noise8/', n_latent=12, prior_noise_level=8, n_epochs=25, batch_size=256)

train_VAE('VAE_normal', '.,/results/test_model_normal_L6_Noise1/', n_latent=6, prior_noise_level=1, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', '../results/test_model_normal_L6_Noise2/', n_latent=6, prior_noise_level=2, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', '../results/test_model_normal_L6_Noise4/', n_latent=6, prior_noise_level=4, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', '../results/test_model_normal_L6_Noise8/', n_latent=6, prior_noise_level=8, n_epochs=25, batch_size=256)
