import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = torch.matmul(x, self.weight.T)
        x = x / 2
        x = torch.sum(x, dim=1, keepdim=True)
        x = x * self.scaling_factor
        return x