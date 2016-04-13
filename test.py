import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


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




