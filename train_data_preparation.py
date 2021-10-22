# -----------------------------------------------------------------------------------
#   * make dataset of small sample size conditions of SQ
# -----------------------------------------------------------------------------------

import os
import numpy as np
import scipy.io as scio
import pandas as pd
import torch
from torch.utils import data

import math
from params import get_parameters
from utils.ops import make_folder
import data.constants as cte


config = get_parameters()

make_folder(config.results, config.dataset_path, config.version)

dataset_path = os.path.join(config.results, config.dataset_path, config.version)


Aggregate_list = []


def search_file(path, str):
    '''
    文件目录检索
    Aggregate_list:预分配空间，返回绝对路径，列表形式
    path:文件根目录
    str:文件名关键词
    '''
    for file in os.listdir(path):
        this_path = os.path.join(path, file)
        if os.path.isfile(this_path):
            if str in this_path:
                Aggregate_list.append(this_path)
        else:
            search_file(this_path, str)
    return Aggregate_list


def text_read(filename):
    '''
    read .txt file and return matrix(.asarray)
    filename: root directory
    '''
    # Try to read a txt file and return a matrix.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
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


def sample_independent(root, mark, num, size, save=False, overlapping=0, interval=0):

    for i in mark:
        name = 'REC' + str(i) + '_ch2.txt'
        namelist = search_file(root, name)
    print(namelist)

    cover = int(size * overlapping)
    datalen = num * (size - cover + interval) + cover-interval
    datatype = len(mark)

    data = np.zeros((num, size))

    for i in range(datatype):

        data_root = namelist[i]
        rawdata = text_read(data_root)

        if i == datatype-1:

            rawdata = rawdata.squeeze()
            len_ = len(rawdata)
            num_ = math.floor((len_ - cover + interval) /(size - cover + interval))

            samples = np.zeros((num_, size))

            for j in range(0, num_):
                index = j * (size - cover + interval) + np.arange(0, size)

                samples[j, :] = rawdata[index]

            df = pd.DataFrame(list(samples), columns=list(np.arange(0, size)))

            saveroot = dataset_path + f'/nc.csv'
            df.to_csv(path_or_buf=saveroot, index=None)


        else:
            rawdata = rawdata[:datalen,:]
            rawdata = rawdata.squeeze()

            for j in range(0, num):
                internal = j * (size - cover + interval) + np.arange(0, size)

                data[j, :] = rawdata[internal]

            label = np.full((num, 1), i)

            df = pd.DataFrame(list(data), columns=list(np.arange(0, size)))
            df['label'] = list(label.squeeze())

            saveroot = dataset_path + f'/raw_{i}.csv'
            df.to_csv(path_or_buf=saveroot, index=None)





root = 'F:\Works\RD\SQ_bearing'  
small_sample = sample_independent(cte.SQ, cte.SQ_TS,  config.batch_size, config.size, save=True)

print('#######  Sample done!  #######')

