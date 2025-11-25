import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for GELU activation
gelu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_forward_kernel(const float* x, float* y, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float inner = sqrt(2.0f / M_PI) * (xi + 0.044715f * xi * xi * xi);
        y[idx] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

torch::Tensor gelu_forward_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);
    return y;
}
"""

# Compile the CUDA kernel inline
gelu_cpp_source = "torch::Tensor gelu_forward_cuda(torch::Tensor x);"
gelu_extension = load_inline(
    name="gelu_cuda",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_cuda_source,
    functions=["gelu_forward_cuda"],
    verbose=False,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_forward = gelu_extension.gelu_forward_cuda

    def forward(self, x):
        return self.gelu_forward(x)

# Keep the original input functions unchanged
def get_inputs():
    return [torch.rand(batch_size, dim).cuda()]

def get_init_inputs():
    return []

# Define batch size and dim as in the original code
batch_size = 8192
dim = 8192