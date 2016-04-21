import numpy as np
from matplotlib import pyplot as plt
from utils import sigmoid, get_cmap, binarize


a = np.array([[0.2, 0.5, 0.8], [0.6, 0.3, 0.2]])
print a
b = binarize(a)
print b


x = np.random.uniform(size=10000)

x = np.random.normal(size=10000)

# the histogram of the data

n, bins, patches = plt.hist(x, 100, normed=1, facecolor='green', alpha=0.75)

plt.xlabel('Pre-activation')
plt.ylabel('Probability')
plt.title('Histogram of Z - Prior Noise 4')
plt.grid(True)

plt.show()


# the histogram of the data

#sigmoidX = sigmoid(8*(x-0.5)+3)

sigmoidX = sigmoid(x*4)

n, bins, patches = plt.hist(sigmoidX, 100, normed=1, facecolor='green', alpha=0.75)

plt.xlabel('Pre-activation')
plt.ylabel('Probability')
plt.title('Histogram of Z - Prior Noise 4')
plt.grid(True)

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)

cmap = get_cmap(10)
colour_array = []
idx_array = np.zeros((10,1))
for s in xrange(10):
	idx_array[s,0] = s+1
	colour_array.append(cmap(s+1))

plt.scatter(idx_array[:,0], idx_array[:,0], color=colour_array)

# for x in idx_array[:,0]:
# 	ax.annotate('%s'%x, x=x, textcoords='data')

plt.grid()
plt.show()

