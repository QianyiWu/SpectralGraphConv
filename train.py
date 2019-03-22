#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wqy
"""
import os
from model.SPGCVAE import SPGCVAE
import argparse

from dataloader.dataloader import *

parser = argparse.ArgumentParser()
parser.add_argument('-g', action='store', dest="gpu", default='3',
                    help="gpu id")
parser.add_argument('-e', action='store', dest="epoch", default=50,
                    help="training epoch")
parser.add_argument('-lr', action='store', dest="l_rate", default=0.0001,
                    help="learning rate")
parser.add_argument('-m', action='store', dest="mode", default= '2moji',
                    help="training mode")
parser.add_argument('-l', action='store_true', dest="load",
                    help="load pretrained model")
parser.add_argument('-t', action='store_false', dest="train",
                    help="use -t to switch to testing step")
parser.add_argument('-s', action='store', dest="suffix",default='',
                    help="suffix of filename")
parser.add_argument('-p', action='store', dest="test_people_id",default=141,
                    help="id of test people")
parser.add_argument('-b', action='store', dest="batch_size",default=2,
                    help="batch_size")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
epoch = int(args.epoch)
l_rate = float(args.l_rate)
load = args.load
train = args.train
mode = args.mode
suffix = args.suffix
people_id = int(args.test_people_id)
# filename prefix
prefix = mode
batch_size = args.batch_size

input_dim = 4525*9
output_dim = 4525*9
feature_dim = 9
prefix = 'mery'
net = SPGCVAE(input_dim, output_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=batch_size, MAX_DEGREE=2)

dataloader = DataLoader('mery')
dataset_info = dataloader.dataset_info()
n_data = dataset_info[0]
print('Dataset {} including {} data, each data has size of {}'.format(prefix, n_data, dataset_info[1:]))

current_log = []
for i in range(epoch):
    log = np.zeros((epoch, ))
    for batch, (input_data, target_data) in enumerate(dataloader.load_data(batch_size)):
        err_re, err_kl, err_total, err_regular = net.train_func([input_data, target_data])
        if batch%50 == 0:
            print(('Epoch: {:3}, batch: {:3}').format(i, batch))
            print(('total loss: {:8.6f}, rec_loss: {:8.6f}, kl_loss: {:8.6f}').format(err_total, err_re, err_kl))
        log[i]+= err_total
    log[i]/= n_data
    current_log.append(log[i])
    # save training plot
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    plt.plot(current_log, 'r-', label='train')
    plt.savefig('current_log.png')
    if i%50==0:
        net.save_model(i)


