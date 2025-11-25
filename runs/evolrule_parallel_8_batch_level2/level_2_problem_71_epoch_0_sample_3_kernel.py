import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor_reciprocal = 1.0 / divisor  # Precompute reciprocal for faster multiplication

    def forward(self, x):
        x = self.conv(x)
        x.mul_(self.divisor_reciprocal)  # In-place multiplication to save memory
        x = F.leaky_relu(x, negative_slope=0.01)
        return x