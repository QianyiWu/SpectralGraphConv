import numpy as np
from .data_util import *

class DataLoader(object):
    def __init__(self, dataset_name, input_size = 4525*9, output_dim = 4525*9,  is_train = True, is_normalize = True):
        self.dataset_name = dataset_name
        self.input_size = input_size
        self.output_dim = output_dim
        if is_train:
            self.dataset = np.load(('data/{}/train.npy').format(dataset_name))
        else:
            self.dataset = np.load(('data/{}/test.npy').format(dataset_name))

        if is_normalize:
            M_list = np.load(('data/{}/max_data.npy').format(self.dataset_name))
            m_list = np.load(('data/{}/min_data.npy').format(self.dataset_name))
            normalize_fromfile(self.dataset, M_list, m_list)
        else:
            pass


    def dataset_info(self):
        return self.dataset.shape

    def load_data(self, batch_size = 1, prefix = None, is_normalize = True, is_shuffle = True,):
        self.n_batches = int(self.dataset.shape[0]/batch_size) 

        self.train_data_list = [i for i in range(self.n_batches)]
        np.random.shuffle(self.train_data_list)
        for i in range(self.n_batches - 1):
            index = self.train_data_list[i]
            input_data = self.dataset[index*batch_size: (index+1)*batch_size]
            target_data = self.dataset[index*batch_size: (index+1)*batch_size]
            yield input_data, target_data











