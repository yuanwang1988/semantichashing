import numpy as np
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt


#data_path = './Autoencoder/MNIST_autoencoder_784_392_196_98_tanh_True_4_data.npz'
#data_path = './Autoencoder/MNIST_autoencoder_784_392_196_98_49_tanh_True_4_data.npz'
#data_path = './Autoencoder/MNIST_autoencoder_784_392_196_98_49_20_tanh_True_4_data.npz'
#data_path = './Autoencoder/MNIST_autoencoder_784_392_196_98_49_24_12_tanh_True_4_data.npz'
#data_path = './Autoencoder/MNIST_autoencoder_784_392_196_98_49_24_12_6_tanh_True_4_data.npz'

#data_path = './VariationalAutoencoder-Modified/VAE_normal_tanh_L49_N4_data.npz'
#data_path = './VariationalAutoencoder-Modified/VAE_normal_tanh_L20_N4_data.npz'
#data_path = './VariationalAutoencoder-Modified/VAE_normal_tanh_L12_N4_data.npz' 
#data_path = './VariationalAutoencoder-Modified/VAE_normal_tanh_L6_N4_data.npz'

# data_path = './VariationalAutoencoder-Modified/VAE_beta_approx_L49_N10_data.npz'
#data_path = './VariationalAutoencoder-Modified/VAE_beta_approx_L20_N10_data.npz'
#data_path = './VariationalAutoencoder-Modified/VAE_beta_approx_L12_N10_data.npz'
data_path = './VariationalAutoencoder-Modified/VAE_beta_approx_L6_N10_data.npz'


npz_file = np.load(data_path)

X_test = npz_file['X_test']
y_test = npz_file['y_test']
z_test = npz_file['z_test']

y_test_1hot = np.zeros((y_test.shape[0], 10), dtype=int)
for i in xrange(y_test.shape[0]):
	y_test_1hot[i, y_test[i]] = 1

clf = DecisionTreeClassifier(random_state=0)
clf.fit(z_test, y_test_1hot)
importance_score = clf.feature_importances_

sort_idx = np.argsort(-importance_score)

print importance_score[sort_idx]


#plot importance_score
fig, ax = plt.subplots()
ind = np.arange(z_test.shape[1])
width = 0.35

ax.bar(ind, importance_score[sort_idx], color='red')
#ax.set_xticklabels(ind)

plt.xlabel('Feature Index (Sorted)')
plt.ylabel('Feature Importance - Gini')
plt.title('Feature Importance - Gini (Sorted)')

plt.show()