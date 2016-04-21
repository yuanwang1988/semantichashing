from __future__ import division

import numpy as np
import theano
import theano.tensor as T


def relu(x):
    return T.switch(x<0, 0, x)

def betaln(alpha, beta):
    return T.gammaln(alpha)+T.gammaln(beta) - T.gammaln(alpha+beta)

def hard_cap(x, lowerbound, upperbound):
    lower_cap = T.switch(x<lowerbound, lowerbound, x)
    return T.switch(lower_cap>upperbound, upperbound, x)

def rmse_score(x, x_recon):
	return T.mean((x-x_recon)**2)