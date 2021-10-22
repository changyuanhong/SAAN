import torch
import numpy as np
import scipy.io as scio
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import pandas as pd


def scaler(data):
    max = np.max(data, axis=1)
    max = np.reshape(max, [-1, 1])
    min = np.min(data, axis=1)
    min = np.reshape(min, [-1, 1])
    return (data - min) / (max - min)*2-1


def get_oridata(fileroot):

    rawdata = pd.read_csv(filepath_or_buffer=fileroot).values
    # 数据分割
    r_data, r_label = np.split(rawdata, 
                               (1024, ),
                               axis=1) 

    data = torch.Tensor(scaler(r_data))
    data = Variable(data)

    return data


def plot_curve(data, saveroot):
    plt.figure()
    plt.subplot(211)

    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, label='K-MMD')
    plt.legend()
    plt.subplot(212)

    x = data[:, 0]
    y = data[:, 2]
    plt.plot(x, y, label='EMD')
    plt.legend()

    plt.savefig(saveroot, bbox_inches='tight')
    # plt.show()


def guassian_kernel(source,
                    target,
                    kernel_mul=2.0,
                    kernel_num=5,
                    fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params: 
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul: 
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
                sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0]) + int(
        target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并

    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)),
                                       int(total.size(1)))

    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)),
                                       int(total.size(1)))

    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1)**2).sum(2)

    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)

    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul**(kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # 高斯核函数的数学表达式
    kernel_val = [
        torch.exp(-L2_distance / bandwidth_temp)
        for bandwidth_temp in bandwidth_list
    ]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul: 
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
                loss: MMD loss
    '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source,
                              target,
                              kernel_mul=kernel_mul,
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


def emd(source, target):
    emd = 0
    for i in range(0, len(source)):
        emd += wasserstein_distance(source[i, :], target[i, :])
    emd = emd / len(source)

    return emd
