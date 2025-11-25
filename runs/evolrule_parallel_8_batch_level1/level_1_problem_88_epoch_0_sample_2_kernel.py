import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        float x_cubed = xi * xi * xi;
        float inner = xi + 0.044715f * x_cubed;
        inner *= 0.7978845608f; // sqrt(2 / M_PI)
        float tanh_val = tanhf(inner);
        out[idx] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for GELU
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = gelu  # Store the loaded module

    def forward(self, x):
        return self.gelu.gelu_cuda(x)

batch_size = 8192
dim = 8192

def get_inputs():
    return [torch.rand(batch_size, dim, device='cuda')]

def get_init_inputs():
    return []