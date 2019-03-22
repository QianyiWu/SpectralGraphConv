#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wqy
"""
import numpy as np
import scipy.sparse as sp
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from .network import *
from .basic_function import *


class SPGCVAE(object):

    def __init__(self, input_dim, output_dim, prefix, lr, load, feature_dim=9, latent_dim=25, kl_weight=0.000005, batch_size=1, MAX_DEGREE=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.load = load
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.v = int(input_dim / feature_dim)
        self.out_v = int(output_dim / feature_dim)
        self.prefix = prefix
        self.hidden_dim = 300
        self.lr = lr
        self.M_list = np.load(('data/{}/max_data.npy').format(self.prefix))
        self.m_list = np.load(('data/{}/min_data.npy').format(self.prefix))
        self.kl_weight = kl_weight
        self.batch_size = batch_size
        self.build_model(MAX_DEGREE)

    def build_model(self, MAX_DEGREE):
        SYM_NORM = True
        A = sp.load_npz(('data/{}/{}_adj_matrix.npz').format(self.prefix, self.prefix))
        L = normalized_laplacian(A, SYM_NORM)
        T_k = chebyshev_polynomial(rescale_laplacian(L), MAX_DEGREE)
        support = MAX_DEGREE + 1
        self.encoder, self.decoder, self.gcn_vae = network(T_k, support, batch_size=self.batch_size, \
                                                                          feature_dim=self.feature_dim, v=self.v, \
                                                                          input_dim=self.input_dim, output_dim = self.output_dim)
         
        self.set_loss()

        if self.load:
            self.load_models()
        else:
            # self.encoder.load_weights('model/encoder_exp_people.h5')
            pass

    def set_loss(self):
        '''
        define of loss function,
        including rec_loss: reconstruction loss
                  kl_loss: KL-divergence loss
                  regular loss: regularization loss
        '''
        # rec loss
        self.output = Input(shape=(self.output_dim,))
        real = self.gcn_vae.get_input_at(0)
        ratio = K.variable(self.M_list - self.m_list)
        s = K.variable(self.m_list + self.M_list)

        self.rec_loss = K.mean(K.abs((self.output - self.gcn_vae(real)) * ratio )) / 1.8 
       
        # regular loss
        weights = self.gcn_vae.trainable_weights
        self.regular_loss = 0
        for w in weights:
            self.regular_loss += 0.000002 * K.sum(K.square(w))

        # kl_loss
        z_mean, z_log_var, z = self.encoder(real)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        self.kl_loss = self.kl_weight * K.mean(-0.5*K.sum(kl_loss, axis = -1))

        self.loss = self.rec_loss + self.kl_loss + self.regular_loss

        training_updates = (Adam(lr=self.lr)).get_updates(weights, [], self.loss)
        self.train_func = K.function([real, self.output], [self.rec_loss, self.kl_loss, self.loss,  self.regular_loss], training_updates)
        self.test_func = K.function([real, self.output], [self.rec_loss, self.kl_loss, self.loss])

    def save_model(self, suffix = None):
        import os
        if os.path.exists('result'):
            pass
        else:
            os.mkdir('result')
        if suffix == None:
            self.encoder.save_weights('result/encoder.h5')
            self.decoder.save_weights('result/decoder.h5')
        else:
            self.encoder.save_weights(('result/encoder_{}.h5').format(suffix))
            self.decoder.save_weights(('result/decoder_{}.h5').format(suffix))


    def load_model(self):
        self.encoder.load_weights('result/encoder.h5')
        self.decoder.load_weights('result/decoder.h5')

