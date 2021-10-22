import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')
sns.plotting_context("paper")

class Visual(object):

    def __init__(self, file):
        self.file = file

    def load_data(self, sheet):

        rawdata = pd.ExcelFile(self.file).parse(sheet, dtype='object')
        res = []

        for i in range(rawdata.shape[0]):

            acc_temp = rawdata.iloc[i, 1:].values.reshape(-1, 1)

            index1 = rawdata.loc[i, ['iter']].values
            index1 = np.expand_dims(index1, 0).repeat(
                acc_temp.shape[0], axis=0)

            index2 = np.arange(acc_temp.shape[0]).reshape(-1, 1)
            num = np.hstack((index1, acc_temp, index2))

            res.extend(num)

        res = np.asarray(res)

        res = pd.DataFrame(list(res), columns=['iter', 'acc', 'number'])

        res = res.dropna(axis=0, how='any')
        res = res.reset_index(drop=True)

        return res

    def plot_acc(self, sheet):

        rawdata = self.load_data(sheet)

        plt.figure(figsize=(8, 6))
        ax = sns.lineplot(x=rawdata.columns[0], y=rawdata.columns[1],
                          markers=True, dashes=False, data=rawdata)
        
        plt.show()