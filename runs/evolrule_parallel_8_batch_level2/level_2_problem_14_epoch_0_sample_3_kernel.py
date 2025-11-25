import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Compute column sums of the weight matrix
        col_sums = self.weight.sum(dim=0).view(-1, 1)
        # Perform matrix-vector multiplication and scaling
        x = x @ col_sums  # (batch_size, 1)
        x = x * (self.scaling_factor / 2.0)
        return x

batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]