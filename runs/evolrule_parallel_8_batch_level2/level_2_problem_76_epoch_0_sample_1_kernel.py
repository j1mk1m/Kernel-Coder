import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused GEMM, add, and ReLU kernel here

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        # Define necessary parameters and load the fused kernel
        # ...
        
    def forward(self, x):
        # Use the fused kernel here
        # ...