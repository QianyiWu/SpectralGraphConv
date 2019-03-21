# -*- coding: utf-8 -*-
"""
Created on Mar 21, 2019

@author: wqy
"""
from keras.layers import Reshape, LeakyRelu, Flatten, Activation, Input, Concatenate, Add
from keras.layers import Dense, Lambda, BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.engine.topology import Layer


def sampling(args):
	'''
	Implements the reparameterization trick in Variational AutoEncoder
	Arguments: 
		args (tensor): mean and log of variance of Q(z|x)
	Returns:
		sampled latent vector (tensor)
	'''
	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * epsilon

def network(T_k, support = 3, batch_size = 1, v = 11510, feature_dim = 9, input_dim = 11510 * 9, output_dim = 9499*9, vis = False, hidden_dim = 300,latent_dim = 25):
    g = [K.variable(_) for _ in T_k]
    def gcn(x):
        supports = list()
        for i in range(support):
            num , v , feature_shape = K.int_shape(x)
            LF = list()
            for j in range(batch_size):
                LF.append(K.expand_dims(K.dot(g[i], K.squeeze(K.slice(x, [j,0,0],[1,v,feature_shape]), axis = 0)), axis=0))
            supports.append(K.concatenate(LF,axis=0)) 
        x = K.concatenate(supports)
        return x
    def GConv(x, input_dim, output_dim, support, active_func = LeakyReLU(alpha=0.1)):
        x = Lambda(gcn, output_shape=(v, support*input_dim))(x)
        x = Dense(output_dim)(x)
        x = active_func(x)
        return x

    inputs = Input(shape=(feature_dim * v, ))
    x = Reshape((v, feature_dim))(inputs)
    # One GCN Layer
    x = GConv(x, feature_dim, 128, support)
    x = GConv(x, 128, 64, support)
    x = GConv(x, 64, 64, support)
    x = GConv(x, 64, 64, support)
    x = GConv(x, 64, 1, support, Activation('tanh'))
    x = Flatten()(x)

    x = Dense(hidden_dim)(x)
    x = LeakyReLU(alpha=0.1)(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    x = Activation('sigmoid')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    encoder = Model(inputs,[z_mean, z_log_var, z], name = 'exp_encoder')
    code = Input(shape=(latent_dim, ))
    x = Dense(hidden_dim)(code)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(output_dim)(x)
    output = Activation('tanh')(x)

    decoder = Model(code, output)
    output = decoder(encoder(inputs)[2])
    gcn_vae = Model(inputs, output)
    if vis:
        gcn_vae.summary()
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.mean(K.abs(K.sum(kl_loss, axis=-1)))
    return kl_loss, encoder, decoder, gcn_vae




