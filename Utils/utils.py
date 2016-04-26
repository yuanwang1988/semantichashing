import numpy as np
from scipy.special import comb
#from sympy.combinatorics.graycode import GrayCode
import itertools

#plotting related
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


# def get_graycode_array(N):
# 	gray_code_generator = GrayCode(N)
# 	gray_code_str_array = list(gray_code_generator.generate_gray())

# 	result = np.zeros((len(gray_code_str_array), N), dtype=int)

# 	for i in xrange(len(gray_code_str_array)):
# 		result[i,:] = np.fromstring(gray_code_str_array[i], dtype='u1') - ord('0')

# 	return result
