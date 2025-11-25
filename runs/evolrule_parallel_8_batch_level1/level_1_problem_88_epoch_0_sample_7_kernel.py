import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

batch_size = 8192
dim = 8192

# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        float x_cubed = xi * xi * xi;
        float term = 0.044715f * x_cubed;
        float inner = xi + term;
        float a = 0.7978845608f * inner;
        float tanh_a = tanhf(a);
        out[idx] = 0.5f * xi * (1.0f + tanh_a);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the inline CUDA code for GELU
gelu_extension = load_inline(
    name="gelu_cuda",
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
        self.gelu = gelu_extension.gelu_cuda

    def forward(self, x):
        return self.gelu(x)

def get_inputs():
    return [torch.rand(batch_size, dim).cuda()]

def get_init_inputs():
    return []