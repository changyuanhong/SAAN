import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import os
# from utils.params import get_parameters
# from utils.ops import make_folder




class Tsne(object):
    def __init__(self, img_path):
        self.img_path = img_path
        # self.fea_root = self.img_path + '/features.csv'

    def plot_tsne(self,fea_root,num_classes,name:str,dataset:str):
        batchsize = 100

        raw_features = pd.read_csv(fea_root)

        raw_features.sort_values(by = 'label',inplace = True)
        features = raw_features.groupby('label').head(batchsize)

        feature = features.iloc[:, 0:-1].values
        label = features.iloc[:, -1].values

        # # Set dif color and shape for dif classes
        m = ['o', 's', '^', 'P', '<', 'v', 'D']
        c = ['black', 'red', 'orange', 'green', 'b', 'darkorchid', 'darkcyan']


        tsne = TSNE(perplexity=30, n_components=2, angle=0.7,
                    init='pca', random_state=None)


        feature_tsne = tsne.fit_transform(feature)

        ft_res = np.concatenate((feature_tsne, label.reshape(-1, 1)), axis=1)

        ft = pd.DataFrame(list(ft_res), columns=list(np.arange(0, 3)))


        plt.figure(figsize=(6, 4))
        plt.rc('font', family='Times New Roman')

        for i in range(num_classes):

            ft_k = ft.iloc[i*batchsize:(i+1)*batchsize, 0:2].values

            plt.scatter(ft_k[:, 0], ft_k[:, 1], s=20, c='',
                marker=m[i], edgecolors=c[i], linewidths=1)


        plt.xlabel('Diamension-1', fontsize=12)
        plt.ylabel('Diamension-2', fontsize=12)


        # -----------------------------------------------------------------------------------
        #   & SQ legend setting
        # -----------------------------------------------------------------------------------
        if dataset == 'SQ':
            plt.legend(['IF_1', 'IF_2', 'IF_3', 'OF_1', 'OF_2', 'OF_3', 'NC'], fontsize=8)


        # -----------------------------------------------------------------------------------
        #   & SQ legend setting
        # -----------------------------------------------------------------------------------
        if dataset == 'SA':
            plt.legend(['NC', 'REF', 'ROF', 'OP', 'OF_1', 'OF_2', 'OF_3'], fontsize=8)


        saveroot = self.img_path + './{}.png'.format(name)
        plt.savefig(saveroot, bbox_inches='tight', dpi=800)
        print('#######  TSNE img saved!  #######')

