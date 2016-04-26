"""
Simple demo with multiple subplots.
"""
import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)


N = 2
M = 3

for i in xrange(N):
	for j in xrange(M):
		frame1=plt.subplot(N, M, i*M+j+1)
		plt.plot(x1, y1, 'ko-')

		frame1.axes.get_xaxis().set_visible(False)
		frame1.axes.get_yaxis().set_visible(False)

plt.show()