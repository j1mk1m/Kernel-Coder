# model_new.py
import torch
import torch.nn as nn
from conv2d import conv2d_cuda
from mish import mish_cuda

class Mish(nn.Module):
    def forward(self, x):
        return mish_cuda(x)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.mish1 = Mish()
        self.mish2 = Mish()

    def forward(self, x):
        x = conv2d_cuda(x, self.conv.weight, stride=self.conv.stride, padding=self.conv.padding)
        x = self.mish1(x)
        x = self.mish2(x)
        return x

batch_size   = 64  
in_channels  = 64  
out_channels = 128  
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]