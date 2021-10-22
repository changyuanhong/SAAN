
import os
import time
import torch
import datetime

import torch.nn as nn
from models.resnet import resnet18, resnet10
from torch.autograd import Variable

# from utils.visual_loss import Visualizer
from utils.eval_index import Evaluation
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from torch.utils import data


class Trainer_classifier(object):
    def __init__(self, data_loader, validloader, config, pretrained_model = None):

        # Data loader
        self.data_loader = data_loader
        self.validloader = validloader

        # exact model and loss
        # self.model = config.model

        # Model hyper-parameters
        self.num_classes = config.num_classes
        self.iteration = config.iteration_classifier

        self.lrC = config.lrC

        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2


        self.pretrained_model = pretrained_model
        self.use_tensorboard = config.use_tensorboard

        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version


        # Path
        self.img_path = os.path.join(config.results, config.image_path, self.version)


        self.model_save_path = os.path.join(config.results, config.model_save_path, self.version)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        print('build_model...')
        self.build_model()


        # Start with trained model
        if self.pretrained_model:
            print('load_pretrained_model...')
            self.load_pretrained_model()

    def train_classifier(self):

        step_per_epoch = len(self.data_loader)
        model_save_step = self.iteration*step_per_epoch

        # Start time
        print('Start   ======  training...')
        start_time = time.time()

        for epoch in range(self.iteration):
            for batch_idx, (r_data, r_labels) in enumerate(self.data_loader):

                r_data = r_data.to(self.device)
                r_labels = r_labels.squeeze().to(self.device)

                output,_ = self.C(r_data)
                loss = self.c_loss(output, r_labels)

                pred = output.data.max(1, keepdim=True)[1]
                correct = pred.eq(
                    r_labels.data.view_as(pred)).cpu().sum().float() / len(r_data)

                self.c_optimizer.zero_grad()
                loss.backward()
                self.c_optimizer.step()

                step = epoch * step_per_epoch + batch_idx

                # Print out log info
                if batch_idx % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    print('Elapsed [{}]\tstep[{}/{}]\tLoss: {:.4f}\tAcc：{:.4f}'.format(
                        elapsed, epoch+1, self.iteration, loss.item(), correct.item()))

                    if self.use_tensorboard:
                        self.writer.add_scalar(
                            'data/loss', loss.item(), (step + 1))
                        self.writer.add_scalar(
                            'data/correct', correct.item(), (step + 1))

                if (step+1) % model_save_step == 0:

                    self.test_classifier(self.validloader)



    def build_model(self):
        self.C = resnet10(num_classes=self.num_classes).to(self.device)

        # Loss and optimizer
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.lrC, betas=(
            self.beta1, self.beta2), weight_decay=self.lr_decay)
        self.c_loss = torch.nn.CrossEntropyLoss()



    def load_pretrained_model(self):
        self.C.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_C.pth'.format(self.pretrained_model))))



# -----------------------------------------------------------------------------------
#   & Test the classifier of diagnosis classification 
# -----------------------------------------------------------------------------------


    def test_classifier(self,testloader):


        # Start time
        print('Start   ======  testing...')
        start_time = time.time()

        loss = 0
        acc = 0
        guess = []
        fact = []
        score = []

        features = []
        with torch.no_grad():

            for t_data, t_label in testloader:
                t_data = t_data.to(self.device)
                t_label = t_label.squeeze().to(self.device)

                output,liner= self.C(t_data)


                pred = output.data.max(1, keepdim=True)[1]

                acc += pred.eq(t_label.data.view_as(pred)).cpu().sum()

                output = output.cpu().detach().numpy()
                score.extend(output)
                liner = liner.cpu().detach().numpy()

                guess.extend(pred.cpu())
                fact.extend(t_label.cpu())
                features.extend(liner)

        guess = np.asarray(guess)
        fact = np.asarray(fact)
        score = np.asarray(score)

        eval_function = Evaluation(self.num_classes)
        #eval_function.roc_curve(fact,score)
        eval_function.index_score(fact,guess)


        # -----------------------------------------------------------------------------------
        #   & Save the features vector of the validation dataset for TSNE using resnet.
        # -----------------------------------------------------------------------------------
        
        ft = pd.DataFrame(features,columns=list(np.arange(0,512)))
        ft['label'] = fact

        ft.sort_values(by = 'label',inplace = True)
        ft_sample = ft.groupby('label').head(100)

        savepath = self.img_path +'./classify.csv'
        ft_sample.to_csv(path_or_buf=savepath,index=None)


        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        print('Elapsed [{}]\nAcc:{}/{}\t({:.2f}%)\n'.format(
            elapsed, acc, len(testloader.dataset), 100*acc.float()/len(testloader.dataset)))
        self.plot_confusion_matrix(guess, fact)

    def plot_confusion_matrix(self, guess, fact, savepic=True):

        
        cnf_matrix = confusion_matrix(fact, guess)
        print(cnf_matrix)


        # -----------------------------------------------------------------------------------
        #   & calculate f1-score for each health condition
        # -----------------------------------------------------------------------------------
        pn = np.sum(cnf_matrix, axis=0)
        tf = np.sum(cnf_matrix, axis=1)

        diag = np.diag(cnf_matrix)
        p = diag / pn
        r = diag / tf
        score = np.around(2 * p * r /(p + r),decimals=3)
        
        print(score)


        x = np.sum(cnf_matrix, axis=1)
        x = np.reshape(x, [-1, 1])
        cnf_matrix = cnf_matrix / x
        cnf_matrix = np.around(cnf_matrix, decimals=2)
        
        plt.figure(figsize=(6,4))

        plt.rc('font',family='Times New Roman')
        plt.imshow(cnf_matrix, cmap=plt.cm.Blues)  # Create the basic matrix.

        # Add title and Axis Labels
        # plt.title('Confusion Matrix')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        # Add appropriate Axis Scales
        class_names = set(fact)  # Get class labels to add to matrix
        tick_marks = np.arange(len(class_names))

        # dataset = 'sa'

        # if dataset == 'cw':
        #     label_ = ['IRF', 'RF', 'ORF', 'NC']
        # elif dataset == 'sq':
        #     label_ = ['IRF_1', 'IRF_2', 'IRF_3', 'ORF_1', 'ORF_2', 'ORF_3', 'NC']
        # elif dataset == 'sa':
        #     label_ = ['NC', 'REF', 'ROF', 'ORP', 'ORF_1', 'ORF_2', 'ORF_3']


        # plt.xticks(tick_marks, label_, fontsize=9, rotation=45)
        # plt.yticks(tick_marks, label_, fontsize=9, rotation=45)


        # -----------------------------------------------------------------------------------
        #   & number label
        # -----------------------------------------------------------------------------------
        # plt.xticks(tick_marks, class_names, fontsize=12)
        # plt.yticks(tick_marks, class_names, fontsize=12)

        # Add Labels to Each Cell
        thresh = cnf_matrix.max() / 2.  # Used for text coloring below
        # Here we iterate through the confusion matrix and append labels to our visualization.
        for i, j in itertools.product(range(cnf_matrix.shape[0]),
                                      range(cnf_matrix.shape[1])):
            plt.text(j,
                     i,
                     cnf_matrix[i, j],
                     horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black",
                     fontsize=12)

        # Add a Side Bar Legend Showing Colors
        # plt.colorbar()

        # if savepic:
        #     saveroot = self.img_path + '/matrix.png'
        #     plt.savefig(saveroot, bbox_inches='tight', dpi=800)
