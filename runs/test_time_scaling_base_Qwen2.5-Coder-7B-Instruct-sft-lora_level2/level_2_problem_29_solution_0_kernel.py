import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mish = mish

    def forward(self, x):
        x = self.linear(x)
        x = self.mish.mish_cuda(x)
        x = self.mish.mish_cuda(x)
        return x