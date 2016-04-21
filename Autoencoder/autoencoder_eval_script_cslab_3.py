from autoencoder_eval_script import eval_autoencoder_hashlookup_precision_recall


eval_autoencoder_hashlookup_precision_recall('MNIST_autoencoder_784_392_196_98_49_20_tanh', './results/final_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_1', noise_flag = True, noise_level=1, Limit = 2500, visual_flag = False)
eval_autoencoder_hashlookup_precision_recall('MNIST_autoencoder_784_392_196_98_49_20_tanh', './results/final_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_2', noise_flag = True, noise_level=2, Limit = 2500, visual_flag = False)
eval_autoencoder_hashlookup_precision_recall('MNIST_autoencoder_784_392_196_98_49_20_tanh', './results/final_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_8', noise_flag = True, noise_level=8, Limit = 2500, visual_flag = False)
eval_autoencoder_hashlookup_precision_recall('MNIST_autoencoder_784_392_196_98_49_20_tanh', './results/final_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_16', noise_flag = True, noise_level=16, Limit = 2500, visual_flag = False)
