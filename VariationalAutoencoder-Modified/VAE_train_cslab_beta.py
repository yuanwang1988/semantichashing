import sys
sys.path.append('../')

from VAE_train_script import train_VAE

train_VAE('VAE_beta_approx', './results/test_model_beta_L49_Noise1/', n_latent=49, prior_noise_level=1, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L49_Noise2/', n_latent=49, prior_noise_level=2, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L49_Noise10/', n_latent=49, prior_noise_level=10, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L49_Noise20/', n_latent=49, prior_noise_level=20, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L49_Noise100/', n_latent=49, prior_noise_level=100, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L49_Noise200/', n_latent=49, prior_noise_level=200, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L49_Noise100/', n_latent=49, prior_noise_level=1000, n_epochs=25, batch_size=256)

train_VAE('VAE_beta_approx', './results/test_model_beta_L20_Noise1/', n_latent=20, prior_noise_level=1, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L20_Noise2/', n_latent=20, prior_noise_level=2, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L20_Noise10/', n_latent=20, prior_noise_level=10, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L20_Noise20/', n_latent=20, prior_noise_level=20, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L20_Noise100/', n_latent=20, prior_noise_level=100, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L20_Noise200/', n_latent=20, prior_noise_level=200, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L20_Noise1000/', n_latent=20, prior_noise_level=1000, n_epochs=25, batch_size=256)

