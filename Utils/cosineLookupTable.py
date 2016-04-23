import numpy as np
from scipy.special import comb
from scipy.spatial import distance
import itertools


#########################################
# Public Functions:						#
#########################################

class cosineLookupTable(object):
	def __init__(self, Z = None, X = None):
		if Z == None and X == None:
			self.dataZ = np.array([])
			self.dataX = np.array([])
		elif Z != None and X != None:
			self.dataZ = Z
			self.dataX = X
		else:
			print 'Error! X and Z do not match'

	def add(self, Z, X):
		self.dataZ = np.vstack(self.dataZ, key)
		self.dataX = np.vstack(self.dataX, val)

	def lookup(self, key):
		cosineDist = np.zeros(self.dataZ.shape[0])
		for i in xrange(self.dataZ.shape[0]):
			cosineDist[i] = distance.cosine(self.dataZ[i,:], key)

		sorted_idx = np.argsort(cosineDist)

		return self.dataX[sorted_idx], self.dataZ[sorted_idx]