import sys
sys.path.append('../')

from VAE_train_script import train_VAE

# train_VAE('VAE_beta_approx', './results/test_model_beta_L12_Noise1/', beta_flag=False, n_latent=12, prior_noise_level=1, n_epochs=25, batch_size=256)
# train_VAE('VAE_beta_approx', './results/test_model_beta_L12_Noise2/', beta_flag=False, n_latent=12, prior_noise_level=2, n_epochs=25, batch_size=256)
# train_VAE('VAE_beta_approx', './results/test_model_beta_L12_Noise10/', beta_flag=False, n_latent=12, prior_noise_level=10, n_epochs=25, batch_size=256)
# train_VAE('VAE_beta_approx', './results/test_model_beta_L12_Noise20/', beta_flag=False, n_latent=12, prior_noise_level=20, n_epochs=25, batch_size=256)
train_VAE('VAE_beta_approx', './results/test_model_beta_L12_Noise100/', beta_flag=False, n_latent=12, prior_noise_level=100, n_epochs=25, batch_size=256)
# train_VAE('VAE_beta_approx', './results/test_model_beta_L12_Noise200/', beta_flag=False, n_latent=12, prior_noise_level=200, n_epochs=25, batch_size=256)
# train_VAE('VAE_beta_approx', './results/test_model_beta_L12_Noise1000/', beta_flag=False, n_latent=12, prior_noise_level=1000, n_epochs=25, batch_size=256)

# train_VAE('VAE_beta_approx', './results/test_model_beta_L6_Noise1/', beta_flag=False, n_latent=6, prior_noise_level=1, n_epochs=25, batch_size=256)
# train_VAE('VAE_beta_approx', './results/test_model_beta_L6_Noise2/', beta_flag=False, n_latent=6, prior_noise_level=2, n_epochs=25, batch_size=256)
# train_VAE('VAE_beta_approx', './results/test_model_beta_L6_Noise10/', beta_flag=False, n_latent=6, prior_noise_level=10, n_epochs=25, batch_size=256)
# train_VAE('VAE_beta_approx', './results/test_model_beta_L6_Noise20/', beta_flag=False, n_latent=6, prior_noise_level=20, n_epochs=25, batch_size=256)
# train_VAE('VAE_beta_approx', './results/test_model_beta_L6_Noise100/', beta_flag=False, n_latent=6, prior_noise_level=100, n_epochs=25, batch_size=256)
# train_VAE('VAE_beta_approx', './results/test_model_beta_L6_Noise200/', beta_flag=False, n_latent=6, prior_noise_level=200, n_epochs=25, batch_size=256)
# train_VAE('VAE_beta_approx', './results/test_model_beta_L6_Noise1000/', beta_flag=False, n_latent=6, prior_noise_level=1000, n_epochs=25, batch_size=256)

