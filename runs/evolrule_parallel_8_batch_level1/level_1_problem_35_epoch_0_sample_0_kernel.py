import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

custom_ops = load(name='custom_kernels', sources=['custom_kernels.cu'], extra_cflags=['-O3'])

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops.group_norm_forward(x, self.gamma, self.beta, self.num_groups)

def get_inputs():
    batch_size = 112
    features = 64
    dim1 = 512
    dim2 = 512
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [64, 8]  # Example parameters (num_features, num_groups)