import os
import numpy as np

def normalize(data, a = 0.9 ,epsilon = 10e-6):
    num,dim = data.shape
    for i in range(dim):
        M = np.max(data[:,i])
        m = np.min(data[:,i])
        if M-m < epsilon:
            M = M+epsilon
            m = m-epsilon
            
        data[:,i] = 2 * a * (data[:,i]-m) / (M-m) -a
    return data

def save_normalize_list(array, data, suffix = 'mery', a = 0.9 ,epsilon = 10e-6):
    num,dim = data.shape
    M_list = np.zeros_like(array)
    m_list = np.zeros_like(array)
    for i in range(dim):
        M = np.max(data[:,i])
        m = np.min(data[:,i])
        if M-m < epsilon:
            M = M+epsilon
            m = m-epsilon
        M_list[i] = M
        m_list[i] = m
    np.save('Max_list_{}'.format(suffix),M_list)
    np.save('Min_list_{}'.format(suffix),m_list)
    return array


def denormalize_fromfile(array, M_list, m_list, a = 0.9):
    num,dim = array.shape
    for i in range(dim):
        M = M_list[i]
        m = m_list[i]
        array[:, i] = (array[:,i]+a)*(M-m)/(2*a)+m
    return array



def normalize_fromfile(array, M_list, m_list, a = 0.9):
    num,dim = array.shape
    for i in range(dim):
        M = M_list[i]
        m = m_list[i]
        array[:, i] = 2 * a * (array[:,i]-m) / (M-m) -a
    return array

def deduce_mean(tup, data_array):
    data = data_array.copy()
    for i in range(len(tup)-1):
        data[tup[i]:tup[i+1]] -= np.mean(data[tup[i]:tup[i+1]], axis = 0)
        
    return data