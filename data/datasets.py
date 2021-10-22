import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils import data

import numpy as np
import pandas as pd

from data.ops import cw_seg, sq_seg, sa_seg
import data.constants as cte



def scaler(data):
    max = np.max(data, axis=1)
    max = np.reshape(max, [-1, 1])
    min = np.min(data, axis=1)
    min = np.reshape(min, [-1, 1])
    return (data - min) / (max - min)*2-1


# -----------------------------------------------------------------------------------
#   & define dataloader of CWRU, SQ, SA dataset 
# -----------------------------------------------------------------------------------

def load_raw(dataset, root, name, fs, size , batchsize):
    
    if dataset == 'sq':
        rawdata = sq_seg(root, name, fs, size)
    elif dataset == 'cw':
        rawdata = cw_seg(root, name, fs, size)
    elif dataset == 'sa':
        rawdata = sa_seg(root, name, fs, size)

    rawdata = rawdata.values
    r_data, r_label = np.split(rawdata,
                               (size, ),
                               axis=1)

    r_data = scaler(r_data)

    r_label = r_label.astype(int)

    r_data = r_data[:, np.newaxis]
    r_data = torch.tensor(r_data, dtype=torch.float32)
    r_label = torch.tensor(r_label, dtype=torch.long)
    dataset = data.TensorDataset(r_data, r_label)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchsize,
                                             shuffle=True)
    return dataloader


# -----------------------------------------------------------------------------------
#   & define dataloader of the combination of real samples and generated samples 
# -----------------------------------------------------------------------------------


def load_sample(root, num_classes, fs, num, batchsize=145, size=1064):

    n_sample = (num + fs) * num_classes

    r_data = np.zeros((n_sample, size))
    r_label = np.zeros((n_sample, 1))

    for i in range(num_classes):

        fileroot = root + f'/G_{i}.csv'

        gen_data = pd.read_csv(filepath_or_buffer=fileroot).values

        fileroot = root + f'/raw_{i}.csv'
        fs_data = pd.read_csv(filepath_or_buffer=fileroot).values

        rawdata = np.concatenate((gen_data, fs_data), axis=0)

        t_data, t_label = np.split(rawdata,
                                   (size, ),
                                   axis=1)

        t_data = scaler(t_data)
        
        t_label = t_label.astype(int)

        for j in range(num+fs):
            r_data[j*num_classes + i, :] = t_data[j, :]
            r_label[j*num_classes + i, :] = t_label[j, :]

    r_data = r_data[:, np.newaxis]
    r_data = torch.tensor(r_data, dtype=torch.float32)
    r_label = torch.tensor(r_label, dtype=torch.long)

    dataset = data.TensorDataset(r_data, r_label)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchsize,
                                             shuffle=True)
    return dataloader


def load_fs_data(fileroot, batchsize):

    rawdata = pd.read_csv(filepath_or_buffer=fileroot).values

    r_data, r_label = np.split(rawdata,
                               (1064, ),
                               axis=1)

    r_data = scaler(r_data)
    r_label = r_label.astype(int)

    r_data = r_data[:, np.newaxis]
    r_data = torch.tensor(r_data, dtype=torch.float32)
    r_label = torch.tensor(r_label, dtype=torch.long)
    dataset = data.TensorDataset(r_data, r_label)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batchsize,
                                             shuffle=True)
    return dataloader



# -----------------------------------------------------------------------------------
#   & DEBUG
# -----------------------------------------------------------------------------------

# root =  'F:\Works\RD\SQ_bearing'
# name = [3594, 3616, 3529, 3510, 3491, 3473, 3638]
# num = 6
# size = 5
# batchsize = 6

# rawloader = load_raw('SQ', root, name, num, size, batchsize)
    
# for i,(data,label) in enumerate(rawloader):
#     print(data)




# def load_validation(batchsize=168, size=1024):
#     root = 'D:/Works/Codes/asset/dataset/'

#     num_classes = 7

#     test_sample = 240 * num_classes

#     t_data = np.zeros((test_sample, size))
#     t_label = np.zeros((test_sample, 1))

#     for i in range(num_classes):
#         fileroot = root + f'test_{i}.csv'
#         rawdata = pd.read_csv(filepath_or_buffer=fileroot).values
#         r_data, r_label = np.split(rawdata,
#                                    (size, ),
#                                    axis=1)

#         r_data = scaler(r_data)
#         r_label = r_label.astype(int)

#         for j in range(len(r_data)):
#             t_data[j*num_classes+i, :] = r_data[j, :]
#             t_label[j*num_classes+i, :] = r_label[j, :]

#     t_data = t_data[:, np.newaxis]
#     t_data = torch.tensor(t_data, dtype=torch.float32)
#     t_label = torch.tensor(t_label, dtype=torch.long)

#     dataset_test = data.TensorDataset(t_data, t_label)
#     dataloader = torch.utils.data.DataLoader(dataset_test,
#                                              batch_size=batchsize,
#                                              shuffle=True)
#     return dataloader




# def load_sq(root, name, fs, size, batchsize):

#     rawdata = sq_seg(name, fs, root, size)
#     rawdata = rawdata.values

#     r_data, r_label = np.split(rawdata,
#                                (size, ),
#                                axis=1)

#     r_data = scaler(r_data)
#     r_label = r_label.astype(int)

#     r_data = r_data[:, np.newaxis]
#     r_data = torch.tensor(r_data, dtype=torch.float32)
#     r_label = torch.tensor(r_label, dtype=torch.long)
#     dataset = data.TensorDataset(r_data, r_label)
#     dataloader = torch.utils.data.DataLoader(dataset,
#                                              batch_size=batchsize,
#                                              shuffle=True)
#     return dataloader


# def load_hr(root, name, fs, size, batchsize):

#     rawdata = hr_seg(name, fs, root, size)
#     rawdata = rawdata.values

#     r_data, r_label = np.split(rawdata,
#                                (size, ),
#                                axis=1)

#     r_data = scaler(r_data)
#     r_label = r_label.astype(int)

#     r_data = r_data[:, np.newaxis]
#     r_data = torch.tensor(r_data, dtype=torch.float32)
#     r_label = torch.tensor(r_label, dtype=torch.long)
#     dataset = data.TensorDataset(r_data, r_label)
#     dataloader = torch.utils.data.DataLoader(dataset,
#                                              batch_size=batchsize,
#                                              shuffle=True)
#     return dataloader
