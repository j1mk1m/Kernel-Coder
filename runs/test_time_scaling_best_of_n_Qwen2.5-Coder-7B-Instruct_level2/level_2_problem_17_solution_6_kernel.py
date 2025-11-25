import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for instance normalization
instance_norm_source = """
// Custom CUDA kernel implementation for instance normalization
"""

instance_norm_cpp_source = (
    "void instance_norm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, float eps);"
)

# Compile the inline CUDA code for instance normalization
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm_mean_var = instance_norm_mean_var

    def forward(self, x):
        x = self.conv(x)
        mean, var = self.instance_norm_mean_var(x)
        x = instance_norm_cuda(x, mean, var, eps=1e-5)
        return x