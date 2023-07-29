"""
decoder
"""


#import functions

import numpy as np
from torch import nn
import torch
import torchvision
from einops import rearrange, reduce
from argparse import ArgumentParser
from pytorch_lightning import LightningModule, Trainer, Callback
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

class Decoder(nn.Module):
    def __init__(self, kernel_size=3, n_filters=64, feature_dim=1024, output_size=32, output_channels=3):
        super().__init__()
        self.init_size = output_size // 2**2 
        self.fc1 = nn.Linear(feature_dim, self.init_size**2 * n_filters)
        # output size of conv2dtranspose is (h-1)*2 + 1 + (kernel_size - 1)
        self.conv1 = nn.ConvTranspose2d(n_filters, n_filters//2, kernel_size=kernel_size, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(n_filters//2, n_filters//4, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(n_filters//4, n_filters//4, kernel_size=kernel_size, padding=1)
        self.conv4 = nn.ConvTranspose2d(n_filters//4, output_channels, kernel_size=kernel_size+1)
        
    def forward(self, x):
        B, _ = x.shape
        y = self.fc1(x)
        y = rearrange(y, 'b (c h w) -> b c h w', b=B, h=self.init_size, w=self.init_size)
        y = nn.ReLU()(self.conv1(y))
        y = nn.ReLU()(self.conv2(y))
        y = nn.ReLU()(self.conv3(y))
        y = nn.Sigmoid()(self.conv4(y))

        return y