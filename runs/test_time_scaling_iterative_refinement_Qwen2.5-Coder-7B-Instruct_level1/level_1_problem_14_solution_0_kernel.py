import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matrix multiplication and triu operation
fused_matmul_triu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Implement the fused kernel here...
"""

fused_matmul_triu_cpp_source = (
    "torch::Tensor fused_matmul_triu_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code for fused matrix multiplication and triu operation
fused_matmul_triu = load_inline(
    name="fused_matmul_triu",
    cpp_sources=fused_matmul_triu_cpp_source,
    cuda_sources=fused_matmul_triu_source,
    functions=["fused_matmul_triu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return fused_matmul_triu.fused_matmul_triu_cuda(A, B)