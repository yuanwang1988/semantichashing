from VAE_train_script import train_VAE

train_VAE('VAE_normal', './results/test_model_normal_L20_Noise4/', n_latent=20, prior_noise_level=4, n_epochs=25, batch_size=256)
train_VAE('VAE_normal_tanh', './results/test_model_normal_tanh_L20_Noise4/', n_latent=20, prior_noise_level=4, n_epochs=25, batch_size=256)
train_VAE('VAE_uniform_tanh', './results/test_model_uniform_tanh_L20_Noise4/', n_latent=20, prior_noise_level=4, n_epochs=25, batch_size=256)