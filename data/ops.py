import os
import scipy.io as scio

import numpy as np
import pandas as pd

import torch
from torch.utils import data


Aggregate_list = []


def search_file(path, str):
    '''
    File directory retrieval：return root directory
    Params:
    Aggregate_list: Preallocated space(list)
    path: file root directory
    str: key word of file name
    '''
    for file in os.listdir(path):
        this_path = os.path.join(path, file)
        if os.path.isfile(this_path):
            if str in this_path:
                Aggregate_list.append(this_path)
        else:
            search_file(this_path, str)
    return Aggregate_list


def txt_read(filename):
    '''read file (txt) and return matrix (asarray)
    Params:
    filename: root directory
    '''
    # Try to read a txt file and return a matrix.Return [] if there was a mistake.
    try:
        file = open(filename, 'r', encoding='UTF-8')
    except IOError:
        error = []
        return error
    content = file.readlines()

    rows = len(content)  # 文件行数
    datamat = np.zeros((rows - 16, 1))  # 初始化数组
    row_count = 0

    for i in range(16, rows):  # 从17行开始读取，文件所决定
        content[i] = content[i].strip().split('\t')
        datamat[row_count, 0] = content[i][1]  # 获取第i行2列数据
        row_count += 1

    file.close()
    return datamat


def scaler(data):
    max = np.max(data, axis=1)
    max = np.reshape(max, [-1, 1])
    min = np.min(data, axis=1)
    min = np.reshape(min, [-1, 1])
    return (data - min) / (max - min)*2-1


def scaler_one(data):
    max = np.max(data)
    min = np.min(data)
    return (data - min) / (max - min)*2-1


# -----------------------------------------------------------------------------------
#   & CWRU dataset segment
# -----------------------------------------------------------------------------------

def cw_seg(root, mark, num, size, save=False,  overlapping=0.3, interval=0):
    '''
    Params:
    # & mark: key word of file name (list)
    # & num: sample size
    # & root: file need to search
    # & size: length of sample
    # & interval: the gap of dif samples. default: 200
    '''
    for i in mark:
        name = str(i).zfill(3) + '.mat'
        namelist = search_file(root, name)

   ####### 数据分割 #######
    cover = int(size * overlapping)
    datalen = num * (size - cover + interval) + cover - interval
    datatype = len(mark)

    data = np.zeros((num, size))
    t_data = []
    t_label = []

    for i in range(datatype):

        data_root = namelist[i]

        temp = scio.loadmat(data_root)

        r_data = temp['X' + str(mark[i]).zfill(3) + '_DE_time']


        # ? Normalization or not
        # rawdata = scaler_one(rawdata.squeeze())
        rawdata = r_data.squeeze()


        if datalen > rawdata.shape[0]:
            raise ValueError

        for j in range(0, num):

            index = j * (size-cover+interval) + np.arange(0, size)

            data[j, :] = rawdata[index]
            label = np.full((num, 1), i)

        temp = data
        data = np.zeros((num, size))

        t_data.extend(temp)
        t_label.extend(label)

    t_data = np.asarray(t_data)
    t_label = np.asarray(t_label)

    t_data = np.around(t_data, decimals=4)

    df = pd.DataFrame(list(t_data), columns=list(np.arange(0, size)))
    df['label'] = list(t_label.squeeze())

    if save:
        savepath = './test.csv'
        df.to_csv(path_or_buf=savepath, index=None)
    return df



# -----------------------------------------------------------------------------------
#   & SQ dataset segment
# -----------------------------------------------------------------------------------

def sq_seg(root, mark, num, size, save=False,  overlapping=0, interval=100):

    for i in mark:
        name = 'REC' + str(i) + '_ch2.txt'
        namelist = search_file(root, name)

    cover = int(size * overlapping)
    datalen = num * (size-cover+interval) + cover-interval

    datatype = len(mark)

    data = np.zeros((num, size))
    t_data = []
    t_label = []

    for i in range(datatype):

        data_root = namelist[i]
        rawdata = txt_read(data_root)

        # ? Normalization or not

        # rawdata = scaler_one(rawdata.squeeze())
        rawdata = rawdata.squeeze()

        if datalen > rawdata.shape[0]:
            raise ValueError

        for j in range(0, num):
            index = j * (size-cover+interval) + np.arange(0, size)
            data[j, :] = rawdata[index]
            label = np.full((num, 1), i)

        temp = data
        data = np.zeros((num, size))
        t_data.extend(temp)
        t_label.extend(label)

    t_data = np.asarray(t_data)
    t_label = np.asarray(t_label)

    t_data = np.around(t_data, decimals=4)

    df = pd.DataFrame(list(t_data), columns=list(np.arange(0, size)))
    df['label'] = list(t_label.squeeze())


    if save:
        savepath = './test.csv'
        df.to_csv(path_or_buf=savepath, index=None)
    return df

# -----------------------------------------------------------------------------------
#   & SA dataset segment
# -----------------------------------------------------------------------------------

def sa_seg(root, mark, num, size, save=False,  overlapping=0.1, interval=0):

    for i in mark:
        name = 'F500_' + str(i).zfill(4) + '.mat'
        namelist = search_file(root, name)

    cover = int(size * overlapping)
    datalen = num * (size - cover + interval) + cover - interval
    datatype = len(mark)

    data = np.zeros((num, size))
    t_data = []
    t_label = []

    for i in range(datatype):

        data_root = namelist[i]

        temp = scio.loadmat(data_root)

        rawdata = temp['F500_' + str(mark[i]).zfill(4)]

        rawdata = scaler_one(rawdata.squeeze())

        if datalen > rawdata.shape[0]:
            raise ValueError

        for j in range(0, num):

            index = j * (size-cover+interval) + np.arange(0, size)

            data[j, :] = rawdata[index]
            label = np.full((num, 1), i)

        temp = data
        data = np.zeros((num, size))

        t_data.extend(temp)
        t_label.extend(label)

    t_data = np.asarray(t_data)
    t_label = np.asarray(t_label)

    t_data = np.around(t_data, decimals=4)

    df = pd.DataFrame(list(t_data), columns=list(np.arange(0, size)))
    df['label'] = list(t_label.squeeze())

    if save:
        savepath = './test.csv'
        df.to_csv(path_or_buf=savepath, index=None)
    return df

# -----------------------------------------------------------------------------------
#   & DEBUG
# -----------------------------------------------------------------------------------

# root = 'F:\Works\RD\SQ_bearing'
# name = [1, 1001, 2001, 3001, 4001, 5001, 6001]

# rawdata = sa_seg(root,name, 2, 10)

# print(rawdata)





# # -----------------------------------------------------------------------------------
# #   & 华山数据集划分
# # -----------------------------------------------------------------------------------

# def hr_seg(mark, num, root, size, save=False,  overlapping=0, interval=200):
#     '''
#     Params:
#     mark: key word of file name (list)
#     num: sample size
#     root: file need to search
#     size: length of sample
#     interval: the gap of dif samples. default: 200
#     '''
#     for i in mark:
#         name = 'REC' + str(i).zfill(4) + '_ch3.txt'
#         namelist = search_file(root, name)

#    ####### 数据分割 #######
#     cover = int(size * overlapping)
#     datalen = num * (size-cover+interval) + cover-interval
#     datatype = len(mark)

#     data = np.zeros((num, size))
#     t_data = []
#     t_label = []

#     for i in range(datatype):

#         data_root = namelist[i]
#         rawdata = txt_read(data_root)
#         rawdata = rawdata.T.squeeze()

#         if datalen > rawdata.shape[0]:
#             raise ValueError

#         for j in range(0, num):

#             index = j * (size-cover+interval) + np.arange(0, size)
            
#             data[j, :] = rawdata[index]
#             label = np.full((num, 1), i)

#         temp = data
#         data = np.zeros((num, size))

#         t_data.extend(temp)
#         t_label.extend(label)

#     t_data = np.asarray(t_data)
#     t_label = np.asarray(t_label)

#     t_data = np.around(t_data, decimals=4)

#     df = pd.DataFrame(list(t_data), columns=list(np.arange(0, size)))
#     df['label'] = list(t_label.squeeze())

#     if save:
#         savepath = './test.csv'
#         df.to_csv(path_or_buf=savepath, index=None)
#     return df









# def scaler_torch(data):
#     max = torch.max(data, axis=1)[0]
#     max = max.view(-1, 1)
#     min = torch.min(data, axis=1)[0]
#     min = min.view(-1, 1)
#     return (data - min) / (max - min)*2-1
