import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define your custom CUDA kernels here
custom_kernels_source = """
// Your CUDA kernel definitions go here
"""

custom_kernels_cpp_source = (
    // Your C++ function declarations go here
)

# Compile the inline CUDA code here
custom_kernels = load_inline(
    name="custom_kernels",
    cpp_sources=custom_kernels_cpp_source,
    cuda_sources=custom_kernels_source,
    functions=["your_custom_function"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.custom_kernels = custom_kernels

    def forward(self, x):
        x = self.gemm(x)
        x = self.custom_kernels.your_custom_function(x, self.max_dim)
        x = x - x.mean(dim=1, keepdim=True)
        x = torch.nn.functional.gelu(x)
        return x