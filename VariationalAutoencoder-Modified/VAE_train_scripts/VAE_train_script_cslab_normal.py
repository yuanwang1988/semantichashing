import sys
sys.path.append('../')

from VAE_train_script import train_VAE

train_VAE('VAE_normal', './results/test_model_normal_L49_Noise1/', n_latent=49, prior_noise_level=1, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', './results/test_model_normal_L49_Noise2/', n_latent=49, prior_noise_level=2, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', './results/test_model_normal_L49_Noise4/', n_latent=49, prior_noise_level=4, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', './results/test_model_normal_L49_Noise8/', n_latent=49, prior_noise_level=8, n_epochs=25, batch_size=256)

train_VAE('VAE_normal', './results/test_model_normal_L20_Noise1/', n_latent=20, prior_noise_level=1, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', './results/test_model_normal_L20_Noise2/', n_latent=20, prior_noise_level=2, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', './results/test_model_normal_L20_Noise4/', n_latent=20, prior_noise_level=4, n_epochs=25, batch_size=256)
train_VAE('VAE_normal', './results/test_model_normal_L20_Noise8/', n_latent=20, prior_noise_level=8, n_epochs=25, batch_size=256)
