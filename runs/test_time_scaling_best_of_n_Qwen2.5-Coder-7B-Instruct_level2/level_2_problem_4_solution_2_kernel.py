import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.mish = nn.Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.mish(x)
        x = self.mish(x)
        return x