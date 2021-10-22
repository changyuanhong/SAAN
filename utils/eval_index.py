import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,precision_recall_fscore_support

import pandas as pd


class Evaluation(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def roc_curve(self, label_valid, label_pre):

        label_valid = label_binarize(
            label_valid, classes=list(np.arange(self.num_classes)))

        # 计算每一类的ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(label_valid[:, i], label_pre[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area（方法二）

        fpr["micro"], tpr["micro"], _ = roc_curve(
            label_valid.ravel(), label_pre.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot all ROC curves
        lw = 1.5
        ticksize = 12
        labelsize=14
        legendsize = 12

        plt.figure(figsize=(8,6))
        plt.rc('font',family='Times New Roman')

        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.4f})'
                 ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle='-', linewidth=lw)
        print('********************  ROC-AUC : {:.4f}  ********************'.format(roc_auc["micro"]))

        plt.plot([0, 1], [0, 1], ':', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.grid(axis='y',lw=0.2)

        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.xlabel('False Positive Rate',fontsize=labelsize)
        plt.ylabel('True Positive Rate',fontsize=labelsize)
        plt.legend(loc="lower right")




        record = np.concatenate(
            (fpr["micro"].reshape(-1, 1), tpr["micro"].reshape(-1, 1)), axis=1)

        record = pd.DataFrame(record, columns=['fpr', 'tpr'])
        root = 'F:/Pics/BIMGAN_png/roc_curve.csv'
        record.to_csv(root, index=None)

        root = 'F:/Pics/BIMGAN_png/roc_curve.png'
        plt.savefig(root, bbox_inches='tight', dpi=600)

    def index_score(self, label_valid, label_pre):
        precision = precision_score(label_valid, label_pre, average='macro')
        recall = recall_score(label_valid, label_pre, average='macro')
        f1 = f1_score(label_valid, label_pre, average='macro')
        # print('********************  Index  ********************')
        # print('precision score: {:.4f}'.format(precision))
        # print('recall_index: {:.4f}'.format(recall))
        print('********************  f1_index: {:.4f}  ********************'.format(f1))
        # print('====================  End!  ====================')




