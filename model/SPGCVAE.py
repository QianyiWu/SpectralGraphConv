#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wqy
"""
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from . import network, basic_function


class SPGCVAE(object):

	def __init__(self, input_dim, output_dim, lr, load, feature_dim=9, latent_dim=25, kl_weight=0.000005, batch_size=1, MAX_DEGREE=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prefix = prefix
        self.suffix = suffix
        self.load = load
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.v = int(input_dim / feature_dim)
        self.out_v = int(output_dim / feature_dim)
        self.hidden_dim = 300
        self.lr = lr
        self.kl_weight = kl_weight

        self.batch_size = batch_size
        self.build_model(MAX_DEGREE)

    def build_model(self, MAX_DEGREE):
        SYM_NORM = True
        A = sp.load_npz(('data/{}/{}_adj_matrix.npz').format(self.prefix, self.prefix))
        L = normalized_laplacian(A, SYM_NORM)
        T_k = chebyshev_polynomial(rescale_laplacian(L), MAX_DEGREE)
        support = MAX_DEGREE + 1
        _, self.encoder, self.decoder, self.gcn_vae_exp = get_gcn_vae_exp(T_k, support, batch_size=self.batch_size, \
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
        set up the define of loss function
        '''
        self.target_exp = Input(shape=(self.output_dim,))
        real = self.gcn_vae_exp.get_input_at(0)
        ratio = K.variable(self.M_list - self.m_list)
        s = K.variable(self.m_list + self.M_list)
        # L2 when xyz, L1 when rimd

        self.exp_loss = K.mean(K.abs((self.target_exp - self.gcn_vae_exp(real)) * ratio )) / 1.8 #+ self.away_loss
        

        weights = self.gcn_vae_exp.trainable_weights
        self.regular_loss = 0
        for w in weights:
            self.regular_loss += 0.000002 * K.sum(K.square(w))
        self.loss = self.exp_loss + self.regular_loss

        training_updates = (Adam(lr=self.lr)).get_updates(weights, [], self.loss)
        self.train_func = K.function([real, self.target_exp], [self.exp_loss, self.loss, self.regular_loss], training_updates)
        self.test_func = K.function([real, self.target_exp], [self.exp_loss, self.loss])


