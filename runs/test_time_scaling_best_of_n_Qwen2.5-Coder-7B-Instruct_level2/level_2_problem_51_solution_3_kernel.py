import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels here
gemm_source = """
// Custom CUDA implementation for matrix multiplication (GEMM)
"""

subtract_source = """
// Custom CUDA implementation for subtraction
"""

global_avg_pool_source = """
// Custom CUDA implementation for global average pooling
"""

log_sum_exp_source = """
// Custom CUDA implementation for log-sum-exp operation
"""

gelu_source = """
// Custom CUDA implementation for GELU activation function
"""

residual_add_source = """
// Custom CUDA implementation for residual addition
"""

# Compile the custom CUDA kernels
gemm_module = load_inline(name="gemm", cpp_sources="", cuda_sources=gemm_source, functions=[], verbose=False)
subtract_module = load_inline(name="subtract", cpp_sources="", cuda_sources=subtract_source, functions=[], verbose=False)
global_avg_pool_module = load_inline(name="global_avg_pool", cpp_sources="", cuda_sources=global_avg_pool_source, functions=[], verbose=False)
log_sum_exp_module = load_inline(name="log_sum_exp", cpp_sources="", cuda_sources=log_sum_exp_source, functions=[], verbose=False)
gelu_module = load_inline(name="gelu", cpp_sources="", cuda_sources=gelu_source, functions=[], verbose=False)
residual_add_module = load_inline(name="residual_add", cpp_sources="", cuda_sources=residual_add_source, functions=[], verbose=False)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.original_x = None

    def forward(self, x):
        # Save original input for residual addition
        self.original_x = x.clone().detach()

        # Gemm
        x = self.gemm(x)

        # Subtract
        x = subtract_module.subtract_cuda(x, self.subtract)

        # GlobalAvgPool
        x = global_avg_pool_module.global_avg_pool_cuda(x)

        # LogSumExp
        x = log_sum_exp_module.log_sum_exp_cuda(x)

        # GELU
        x = gelu_module.gelu_cuda(x)

        # ResidualAdd
        x = residual_add_module.residual_add_cuda(x, self.original_x)

        return x