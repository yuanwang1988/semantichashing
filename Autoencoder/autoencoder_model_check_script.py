from autoencoder_eval_script import eval_autoencoder_RMSE, eval_autoencoder_recon

eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_tanh_True_1', noise_flag = True, noise_level=1)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_tanh_True_2', noise_flag = True, noise_level=2)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_tanh_True_8', noise_flag = True, noise_level=8)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_tanh_True_16', noise_flag = True, noise_level=16)

eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_tanh_True_1', noise_flag = True, noise_level=1, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_tanh_True_2', noise_flag = True, noise_level=2, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_tanh_True_8', noise_flag = True, noise_level=8, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_tanh_True_16', noise_flag = True, noise_level=16, nExamples=3)

eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_tanh_True_1', noise_flag = True, noise_level=1)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_tanh_True_2', noise_flag = True, noise_level=2)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_tanh_True_8', noise_flag = True, noise_level=8)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_tanh_True_16', noise_flag = True, noise_level=16)

eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_tanh_True_1', noise_flag = True, noise_level=1, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_tanh_True_2', noise_flag = True, noise_level=2, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_tanh_True_8', noise_flag = True, noise_level=8, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_tanh_True_16', noise_flag = True, noise_level=16, nExamples=3)

eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_20_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_1', noise_flag = True, noise_level=1)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_20_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_2', noise_flag = True, noise_level=2)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_20_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_8',  noise_flag = True, noise_level=8)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_20_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_16',  noise_flag = True, noise_level=16)

eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_20_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_1', noise_flag = True, noise_level=1, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_20_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_2', noise_flag = True, noise_level=2, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_20_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_8',  noise_flag = True, noise_level=8, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_20_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_16',  noise_flag = True, noise_level=16, nExamples=3)

eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_tanh_True_1', noise_flag = True, noise_level=1)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_tanh_True_2', noise_flag = True, noise_level=2)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_tanh_True_8', noise_flag = True, noise_level=8)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_tanh_True_16', noise_flag = True, noise_level=16)

eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_tanh_True_1', noise_flag = True, noise_level=1, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_tanh_True_2', noise_flag = True, noise_level=2, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_tanh_True_8', noise_flag = True, noise_level=8, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_24_12_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_tanh_True_16', noise_flag = True, noise_level=16, nExamples=3)


eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh_True_1', noise_flag = True, noise_level=1)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh_True_2', noise_flag = True, noise_level=2)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh_True_8', noise_flag = True, noise_level=8)
eval_autoencoder_RMSE('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh_True_16', noise_flag = True, noise_level=16)

eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh_True_1', noise_flag = True, noise_level=1, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh_True_2', noise_flag = True, noise_level=2, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh_True_8', noise_flag = True, noise_level=8, nExamples=3)
eval_autoencoder_recon('MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh', './mnist_models/MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh_True_16', noise_flag = True, noise_level=16, nExamples=3)






