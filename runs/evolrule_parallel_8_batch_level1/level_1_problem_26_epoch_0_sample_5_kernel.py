import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

import math

# Define the custom CUDA kernel for GELU tanh approximation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_tanh_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float inner = sqrt(2.f / M_PI) * (xi + 0.044715f * xi * xi * xi);
        out[idx] = 0.5f * xi * (1.f + tanh(inner));
    }
}

torch::Tensor gelu_tanh_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_tanh_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_tanh_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for GELU
gelu_tanh = load_inline(
    name="gelu_tanh",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_tanh = gelu_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_tanh.gelu_tanh_cuda(x.cuda())  # Ensure input is on GPU

# Keep the input generation functions the same as original
def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []