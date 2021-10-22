# -----------------------------------------------------------------------------------
#   * define LOSGAN model
# -----------------------------------------------------------------------------------

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from torch.autograd import Variable
# from torchsummary import summary

import numpy as np
import functools

# import functions
from models.spectral_normalization import SpectralNorm


# -----------------------------------------------------------------------------------
#   & define self-attention layer
# -----------------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.query_conv = SpectralNorm(
            nn.Conv1d(in_channels=in_dim,
                      out_channels=in_dim // 8,
                      kernel_size=1))
        self.key_conv = SpectralNorm(
            nn.Conv1d(in_channels=in_dim,
                      out_channels=in_dim // 8,
                      kernel_size=1))
        self.value_conv = SpectralNorm(
            nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))

        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight_u.data, 0.0, 0.02)
                nn.init.normal_(m.weight_v.data, 0.0, 0.02)
                nn.init.normal_(m.weight_bar.data, 0.0, 0.02)

    def forward(self, input):

        m_batchsize, C, width = input.size()
        proj_query = self.query_conv(input).permute(0, 2, 1)  # B X C X (N)
        proj_key = self.key_conv(input)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(input)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width)

        out = self.gamma * out + input
        return out

# -----------------------------------------------------------------------------------
#   & define generator
# -----------------------------------------------------------------------------------
 
class Generator(nn.Module):
    def __init__(self, z_dim=128, ngf=64, nc=1):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.ngf = ngf
        self.nc = nc
        self.attention = SelfAttention(64)

        self.main = nn.Sequential(
            SpectralNorm(nn.Conv1d(1, 64, 3, 2, 1, bias=False)),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),

            SpectralNorm(nn.Conv1d(64, 128, 3, 2, 1, bias=False)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=False),

            SpectralNorm(nn.Conv1d(128, 256, 3, 2, 1, bias=False)),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=False),

            SpectralNorm(nn.ConvTranspose1d(256, 128, 3, 2, 1,1, bias=False)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=False),

            SpectralNorm(nn.ConvTranspose1d(128, 64, 3, 2, 1,1, bias=False)),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=False),
            
        )
        self.main_2 = nn.Sequential(SpectralNorm(nn.ConvTranspose1d(64, 1, 3, 2, 1, 1,bias=False)))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                
                nn.init.normal_(m.weight_u.data, 0.0, 0.02)
                nn.init.normal_(m.weight_v.data, 0.0, 0.02)
                nn.init.normal_(m.weight_bar.data, 0.0, 0.02)


            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, inputs):
        outputs = self.main(inputs)
        outputs = self.attention(outputs)
        outputs = self.main_2(outputs)
        return outputs




# -----------------------------------------------------------------------------------
#   & define discriminator
# -----------------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.attention = SelfAttention(512)

        self.conv1 = SpectralNorm(nn.Conv1d(nc, ndf, 24, 3, 1, bias=False))
        self.conv2 = SpectralNorm(
            nn.Conv1d(ndf, ndf * 2, 25, 3, 1, bias=False))
        self.conv3 = SpectralNorm(
            nn.Conv1d(ndf * 2, ndf * 4, 20, 3, 1, bias=False))
        self.conv4 = SpectralNorm(
            nn.Conv1d(ndf * 4, ndf * 8, 20, 3, 1, bias=False))
        self.conv5 = SpectralNorm(
            nn.Conv1d(ndf * 8, 1, 5, 1,  bias=False))


        for m in self.modules():
            if isinstance(m, (nn.Conv1d)):
                # nn.init.zeros_(m.weight_u.data)

                nn.init.normal_(m.weight_u.data, 0.0, 0.02)
                nn.init.normal_(m.weight_v.data, 0.0, 0.02)
                nn.init.normal_(m.weight_bar.data, 0.0, 0.02)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        outputs = nn.LeakyReLU(0.2, inplace=False)(self.conv1(inputs))
        outputs = nn.LeakyReLU(0.2, inplace=False)(self.conv2(outputs))
        outputs = nn.LeakyReLU(0.2, inplace=False)(self.conv3(outputs))

        outputs = nn.LeakyReLU(0.2, inplace=False)(self.conv4(outputs))
        outputs = self.attention(outputs)

        outputs = self.conv5(outputs)
        outputs = outputs.view(-1, 1).squeeze()

        return outputs




# -----------------------------------------------------------------------------------
#   & debug (summary)
# -----------------------------------------------------------------------------------
# generator = Generator(z_dim=128)
# fake_image = generator(torch.randn(2, 128))
# print(fake_image.shape)


# dis = Discriminator()
# out = dis(torch.randn(2, 1, 1024))
# print(out.shape)


# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

# model = Generator().to(device)
# summary(model, (128,))

# model = Discriminator().to(device)
# summary(model, (1, 1024))
