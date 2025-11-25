import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float x_cubed = xi * xi * xi;
        float term = 0.044715f * x_cubed;
        float inner = xi + term;
        inner *= 0.7978845608f;  // sqrt(2/pi)
        float tanh_val = tanhf(inner);
        out[idx] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

gelu_cpp_source = """
torch::Tensor gelu_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code for GELU
gelu = load_inline(
    name="gelu",
    cuda_sources=gelu_source,
    cpp_sources=gelu_cpp_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = gelu  # Reference to the custom GELU function

    def forward(self, x):
        return self.gelu.gelu_cuda(x)