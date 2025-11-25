import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x.clone().detach()
        x = self.gemm(x)
        x = x - self.subtract
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = torch.nn.functional.gelu(x)
        x = x + original_x
        return x