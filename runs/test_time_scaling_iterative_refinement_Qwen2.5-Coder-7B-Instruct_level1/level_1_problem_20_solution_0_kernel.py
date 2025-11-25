import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

leaky_relu_module = load(name="leaky_relu_module", sources=["leaky_relu_module.cpp"])

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return leaky_relu_module.leaky_relu_forward_cuda(x, self.negative_slope)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return []