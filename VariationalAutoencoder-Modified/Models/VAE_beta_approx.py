from __future__ import division

import numpy as np
import theano
import theano.tensor as T

import cPickle
from collections import OrderedDict

#custom functions

from theano_utils import relu, betaln, hard_cap, rmse_score

epsilon = 1e-8



class VAE:
    """This class implements the Variational Auto Encoder"""
    def __init__(self, continuous, hu_encoder, hu_decoder, n_latent, x_train, prior_noise_level=4, b1=0.05, b2=0.001, batch_size=100, learning_rate=0.001, lam=0):
        self.continuous = continuous
        self.hu_encoder = hu_encoder
        self.hu_decoder = hu_decoder
        self.n_latent = n_latent
        [self.N, self.features] = x_train.shape

        self.prior_noise_level = prior_noise_level

        self.prng = np.random.RandomState(42)

        self.b1 = np.float32(b1)
        self.b2 = np.float32(b2)
        self.learning_rate = np.float32(learning_rate)
        self.lam = np.float32(lam)

        self.batch_size = batch_size

        sigma_init = 0.01

        create_weight = lambda dim_input, dim_output: self.prng.normal(0, sigma_init, (dim_input, dim_output)).astype(theano.config.floatX)
        create_bias = lambda dim_output: np.zeros(dim_output).astype(theano.config.floatX)

        # encoder
        W_xh = theano.shared(create_weight(self.features, hu_encoder), name='W_xh')
        b_xh = theano.shared(create_bias(hu_encoder), name='b_xh')

        W_halpha = theano.shared(create_weight(hu_encoder, n_latent), name='W_halpha')
        b_halpha = theano.shared(create_bias(n_latent), name='b_halpha')

        W_hbeta = theano.shared(create_weight(hu_encoder, n_latent), name='W_hbeta')
        b_hbeta = theano.shared(create_bias(n_latent), name='b_hbeta')

        # decoder
        W_zh = theano.shared(create_weight(n_latent, hu_decoder), name='W_zh')
        b_zh = theano.shared(create_bias(hu_decoder), name='b_zh')

        self.params = OrderedDict([("W_xh", W_xh), ("b_xh", b_xh), ("W_halpha", W_halpha), ("b_halpha", b_halpha),
                                   ("W_hbeta", W_hbeta), ("b_hbeta", b_hbeta), ("W_zh", W_zh),
                                   ("b_zh", b_zh)])

        if self.continuous:
            W_hxalpha = theano.shared(create_weight(hu_decoder, self.features), name='W_hxalpha')
            b_hxalpha = theano.shared(create_bias(self.features), name='b_hxalpha')

            W_hxsig = theano.shared(create_weight(hu_decoder, self.features), name='W_hxsigma')
            b_hxsig = theano.shared(create_bias(self.features), name='b_hxsigma')

            self.params.update({'W_hxalpha': W_hxalpha, 'b_hxalpha': b_hxalpha, 'W_hxsigma': W_hxsig, 'b_hxsigma': b_hxsig})
        else:
            W_hx = theano.shared(create_weight(hu_decoder, self.features), name='W_hx')
            b_hx = theano.shared(create_bias(self.features), name='b_hx')

            self.params.update({'W_hx': W_hx, 'b_hx': b_hx})

        # Adam parameters
        self.m = OrderedDict()
        self.v = OrderedDict()

        for key, value in self.params.items():
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key)
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key)

        x_train = theano.shared(x_train.astype(theano.config.floatX), name="x_train")

        self.update, self.likelihood, self.encode, self.decode, self.encode_alpha, self.encode_beta, self.eval_rmse = self.create_gradientfunctions(x_train)



    def encoder(self, x):
        h_encoder = T.nnet.relu(T.dot(x, self.params['W_xh']) + self.params['b_xh'].dimshuffle('x', 0))

        alpha = T.dot(h_encoder, self.params['W_halpha']) + self.params['b_halpha'].dimshuffle('x', 0)
        beta = T.dot(h_encoder, self.params['W_hbeta']) + self.params['b_hbeta'].dimshuffle('x', 0)

        alpha = T.nnet.softplus(alpha) + epsilon
        beta = T.nnet.softplus(beta) + epsilon

        return alpha, beta

    def sampler(self, alpha, beta):
        seed = 42

        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        else:
            srng = T.shared_randomstreams.RandomStreams(seed=seed)

        #eps = srng.normal(alpha.shape)

        eps2 = srng.uniform(alpha.shape)

        # Reparametrize

        #eps = hard_cap(eps, -3, 3)

        z = (1-(1-eps2)**(1.0/alpha))**(1.0/beta)

        z = 2*(z-0.5)

        return z

    def decoder(self, x, z):

        #z = T.tanh(z)
        z = z/2.0 + 0.5
        h_decoder = T.nnet.relu(T.dot(z, self.params['W_zh']) + self.params['b_zh'].dimshuffle('x', 0))

        if self.continuous:
            reconstructed_x = T.dot(h_decoder, self.params['W_hxalpha']) + self.params['b_hxalpha'].dimshuffle('x', 0)
            beta_decoder = T.dot(h_decoder, self.params['W_hxsigma']) + self.params['b_hxsigma']

            logpxz = (-(0.5 * np.log(2 * np.pi) + 0.5 * beta_decoder) -
                      0.5 * ((x - reconstructed_x)**2 / T.exp(beta_decoder))).sum(axis=1, keepdims=True)
        else:
            reconstructed_x = T.nnet.sigmoid(T.dot(h_decoder, self.params['W_hx']) + self.params['b_hx'].dimshuffle('x', 0))
            logpxz = - T.nnet.binary_crossentropy(reconstructed_x, x).sum(axis=1, keepdims=True)

        return reconstructed_x, logpxz


    def create_gradientfunctions(self, x_train):
        """This function takes as input the whole dataset and creates the entire model"""
        x = T.matrix("x")

        epoch = T.iscalar("epoch")

        batch_size = x.shape[0]

        alpha, beta = self.encoder(x)
        z = self.sampler(alpha, beta)
        reconstructed_x, logpxz = self.decoder(x,z)

        # Expectation of (logpz - logqz_x) over logqz_x is equal to KLD (see appendix B):
        # KLD = 0.5 * T.sum(1 + beta - alpha**2 - T.exp(beta), axis=1, keepdims=True)

        #KLD = 0.5 * T.sum(1 + beta - (alpha**2 + T.exp(beta)) / (2*(self.prior_noise_level**2)) , axis=1, keepdims=True)

        # KLD = cross-entroy of the sample distribution of sigmoid(z) from the beta distribution
        alpha_prior = 1.0/self.prior_noise_level
        beta_prior = 1.0/self.prior_noise_level
        # sigmoidZ = T.nnet.sigmoid(z)
        # KLD = 25*T.sum((alpha_prior-1)*sigmoidZ + (beta-1)*(1-sigmoidZ) - betaln(alpha_prior,beta), axis=1, keepdims=True)
        # KLD = 0

        KLD = -(betaln(alpha, beta) - betaln(alpha_prior, beta_prior) \
         + (alpha_prior - alpha)*T.psi(alpha_prior) + (beta_prior - beta)*T.psi(beta_prior) \
         + (alpha - alpha_prior + beta - beta_prior)*T.psi(alpha_prior+beta_prior))

        # Average over batch dimension
        logpx = T.mean(logpxz + KLD)
 
        rmse_val = rmse_score(x, reconstructed_x)

        # Compute all the gradients
        gradients = T.grad(logpx, self.params.values())

        # Adam implemented as updates
        updates = self.get_adam_updates(gradients, epoch)

        batch = T.iscalar('batch')

        givens = {
            x: x_train[batch*self.batch_size:(batch+1)*self.batch_size, :]
        }

        # Define a bunch of functions for convenience
        update = theano.function([batch, epoch], logpx, updates=updates, givens=givens)
        likelihood = theano.function([x], logpx)
        eval_rmse = theano.function([x], rmse_val)
        encode = theano.function([x], z)
        decode = theano.function([z], reconstructed_x)
        encode_alpha = theano.function([x], alpha)
        encode_beta = theano.function([x], beta)

        return update, likelihood, encode, decode, encode_alpha, encode_beta, eval_rmse

    def transform_data(self, x_train):
        transformed_x = np.zeros((self.N, self.n_latent))
        batches = np.arange(int(self.N / self.batch_size))

        for batch in batches:
            batch_x = x_train[batch*self.batch_size:(batch+1)*self.batch_size, :]
            transformed_x[batch*self.batch_size:(batch+1)*self.batch_size, :] = self.encode(batch_x)

        return transformed_x

    def save_parameters(self, path):
        """Saves all the parameters in a way they can be retrieved later"""
        cPickle.dump({name: p.get_value() for name, p in self.params.items()}, open(path + "/params.pkl", "wb"))
        cPickle.dump({name: m.get_value() for name, m in self.m.items()}, open(path + "/m.pkl", "wb"))
        cPickle.dump({name: v.get_value() for name, v in self.v.items()}, open(path + "/v.pkl", "wb"))

    def load_parameters(self, path):
        """Load the variables in a shared variable safe way"""
        p_list = cPickle.load(open(path + "/params.pkl", "rb"))
        m_list = cPickle.load(open(path + "/m.pkl", "rb"))
        v_list = cPickle.load(open(path + "/v.pkl", "rb"))

        for name in p_list.keys():
            self.params[name].set_value(p_list[name].astype(theano.config.floatX))
            self.m[name].set_value(m_list[name].astype(theano.config.floatX))
            self.v[name].set_value(v_list[name].astype(theano.config.floatX))

    def get_adam_updates(self, gradients, epoch):
        updates = OrderedDict()
        gamma = T.sqrt(1 - self.b2**epoch) / (1 - self.b1**epoch)

        values_iterable = zip(self.params.keys(), self.params.values(), gradients, 
                              self.m.values(), self.v.values())

        for name, parameter, gradient, m, v in values_iterable:
            new_m = self.b1 * m + (1 - self.b1) * gradient
            new_v = self.b2 * v + (1 - self.b2) * (gradient**2)

            updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v) + epsilon)

            if 'W' in name:
                # MAP on weights (same as L2 regularization)
                updates[parameter] -= self.learning_rate * self.lam * (parameter * np.float32(self.batch_size / self.N))

            updates[m] = new_m
            updates[v] = new_v

        return updates
