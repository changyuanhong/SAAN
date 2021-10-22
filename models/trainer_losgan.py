# -----------------------------------------------------------------------------------
#   * define trainer function of LOSGAN
# -----------------------------------------------------------------------------------

import os
import time
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


# Import local modules
from models.losgan_model import Generator, Discriminator
from models.metric import mmd_rbf, emd


# Visdom visualization
# from utils.visual_loss import Visualizer
# from torchnet import meter



# from torch.utils import data
# import torch.optim as optim

# import argparse
import pandas as pd
# import random
import torch.autograd as autograd
# import scipy.io as io

# # 调用函数
# from data.ops import scaler, scaler_torch
# from utils.params import get_parameters

# from utils.ops import make_folder




class Trainer(object):
    def __init__(self, data_loader, config, pretrained_model=None):

        # Data loader
        self.data_loader = data_loader

        # Model hyper-parameters
        self.version = config.version

        self.z_dim = config.z_dim
        self.size = config.size
        self.lambda_gp = config.lambda_gp

        # Training setting
        self.iteration = config.iteration
        self.batch_size = config.batch_size
        self.num_classes = config.num_classes

        self.lrG = config.lrG
        self.lrD = config.lrD
        self.lr_decay = config.lr_decay

        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = pretrained_model


        self.alpha = config.alpha
        self.beta = config.beta

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Path
        
        self.model_save_path = os.path.join(config.results, config.model_save_path, self.version)

        self.dataset_path = os.path.join(config.results, config.dataset_path, self.version)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        print('build_model...')
        self.build_model()
        '''
        if self.use_tensorboard:
            self.vis = Visualizer(env=config.version)
            
            self.d_loss_meter = meter.AverageValueMeter()
            self.g_loss_meter = meter.AverageValueMeter()
        '''
        # Start with trained model
        if self.pretrained_model:
            print('load_pretrained_model...')
            self.load_pretrained_model()

    def train(self,k):

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model
        else:
            start = 0

        step_per_epoch = len(self.data_loader)
        # model_save_step = self.iteration*step_per_epoch+start
        model_save_step = 200
        # Start time
        print('Start   ======  training...')
        start_time = time.time()

# -----------------------------------------------------------------------------------
#   & Training
# -----------------------------------------------------------------------------------
        fileroot = self.dataset_path + './nc.csv'

        raw_data = pd.read_csv(filepath_or_buffer = fileroot).values

        row_rand_array = np.arange(raw_data.shape[0])
        loss=np.zeros((1500))
        for epoch in range(self.iteration):

            np.random.shuffle(row_rand_array)
            row_rand = raw_data[row_rand_array[0:self.batch_size]]
            row_rand = row_rand[:, np.newaxis]

            for batch_idx, (x_real, _) in enumerate(self.data_loader):

            # -----------------------------------------------------------------------------------
            #   & Updating D
            # -----------------------------------------------------------------------------------

                x_real = x_real.to(self.device)
          

                sx_real = self.D(x_real)

                z_fake = Variable(torch.tensor(row_rand).to(self.device))

                z_star, _ = self.compute_gradient(self.D, self.G, z_fake)

                z_fake_ = z_star.detach()

                x_fake = self.G(z_star)
          

                sx_fake = self.D(x_fake)

            # -----------------------------------------------------------------------------------
            #   & Compute gradient penalty
            # -----------------------------------------------------------------------------------
                alpha = torch.rand(x_real.size(0), 1, 1).to(
                    self.device).expand_as(x_real)
                interpolated = Variable(
                    alpha * x_real + (1 - alpha) * x_fake, requires_grad=True)

                out = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(
                                               out.size()).to(self.device),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 0) ** 2) * self.lambda_gp


                # Backward + Optimize
                self.reset_grad()
                
                d_loss = - torch.mean(sx_real) + torch.mean(sx_fake) + d_loss_gp
                d_loss.backward(retain_graph=True)
                self.d_optimizer.step()

            # -----------------------------------------------------------------------------------
            #   & Computed K-MMD
            # -----------------------------------------------------------------------------------

                sample_g = self.scaler_torch(self.G(z_fake_).squeeze())

                sample_r = self.scaler_torch(x_real.squeeze())

                score_mmd = mmd_rbf(sample_g.detach(), sample_r.detach())
                # score_wd = emd(sample_g.cpu().numpy(),
                #             sample_r.cpu().numpy())
            # -----------------------------------------------------------------------------------
            #   & Updating G
            # -----------------------------------------------------------------------------------


                
                self.reset_grad()

                g_loss = - torch.mean(self.D(self.G(z_fake_))) + score_mmd
                g_loss.backward()

                self.g_optimizer.step()

                step = epoch * step_per_epoch + batch_idx + start



# -----------------------------------------------------------------------------------
#   & Recorder
# -----------------------------------------------------------------------------------
                if self.use_tensorboard:

                    self.d_loss_meter.reset()
                    self.d_loss_meter.add(d_loss.item())


                    self.g_loss_meter.reset()
                    self.g_loss_meter.add(g_loss.item())

                    self.vis.plot_many_stack({'d_loss': self.d_loss_meter.value()[0],
                                              'g_loss': self.g_loss_meter.value()[0]})

                
                # Print out log info
                if batch_idx % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    print('Schedule[{}/{}] Elapsed [{}] \t step[{}/{}] \t DLoss: {:.4f} \t Gloss：{:.4f}'.format(
                        k+1, self.num_classes, elapsed, epoch+1, self.iteration, d_loss.item(), g_loss.item()))
                    loss[epoch]=float(g_loss.item())
                # if (start + epoch+1) % model_save_step == 0 and (start + epoch+1) > 700:
                if (start + epoch+1) % self.iteration == 0:

                    torch.save(self.G.state_dict(),
                               os.path.join(self.model_save_path, 'G_{}_{}.pth'.format(k, start + epoch+1)))
        return loss          
    def evaluate(self, k, niteration, num_sample=200):

        PATH = f'{self.model_save_path}/G_{k}_{niteration}.pth'
        self.G.load_state_dict(torch.load(PATH))

        fileroot = self.dataset_path + './nc.csv'
        raw_data = pd.read_csv(filepath_or_buffer = fileroot).values
        row_rand_array = np.arange(raw_data.shape[0])
        np.random.shuffle(row_rand_array)

        row_rand = raw_data[row_rand_array[0:num_sample]]
        row_rand = row_rand[:, np.newaxis]

        z_fake = torch.tensor(row_rand).type(torch.FloatTensor).to(self.device)


        fake_data = self.G(z_fake).detach().squeeze().cpu().numpy()

        df = pd.DataFrame(list(fake_data), columns=list(np.arange(0, self.size)))

        label = np.full((num_sample, 1), k)
        df['label'] = list(label.squeeze())

        saveroot = self.dataset_path + f'/G_{k}.csv'
        df.to_csv(path_or_buf=saveroot, index=None)

# -----------------------------------------------------------------------------------
#   & Modules
# -----------------------------------------------------------------------------------

    def build_model(self):
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)

        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.lrG, betas=(self.beta1, self.beta2), weight_decay=self.lr_decay)

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.lrD, betas=(self.beta1, self.beta2), weight_decay=self.lr_decay)
     

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()



    def compute_gradient(self, D, G, z):

        z = Variable(z.type(torch.FloatTensor).to(self.device), requires_grad=True)
        y = D(G(z))

        weight = torch.ones(y.size()).to(self.device)

        gradients = autograd.grad(outputs=y,
                                inputs=z,
                                grad_outputs=weight,
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True)[0]

        fem = self.beta + gradients.norm(2, dim=1)**2
        fem = self.alpha / fem
        fem = fem.unsqueeze(1)

        delta = torch.mul(gradients, fem)

        z = delta + z
        z = torch.clamp(z, -1.0, 1.0)
        return z, gradients.norm(2, dim=1)


    def scaler_torch(self, data):
        max = torch.max(data, axis=1)[0]
        max = max.view(-1, 1)
        min = torch.min(data, axis=1)[0]
        min = min.view(-1, 1)
        return (data - min) / (max - min)*2-1
