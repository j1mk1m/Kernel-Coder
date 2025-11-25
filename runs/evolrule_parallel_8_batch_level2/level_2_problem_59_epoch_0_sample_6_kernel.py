import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Swish activation and scaling
swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_scale_kernel(
    const float* x,
    float* out,
    int size,
    float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float sigmoid_xi = 1.0f / (1.0f + expf(-xi));
        out[idx] = xi * sigmoid_xi * scaling_factor;
    }
}

torch::Tensor swish_scale_cuda(torch::Tensor x, float scaling_factor) {
    auto size = x.numel();
    auto out = torch::empty_like(x);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    swish_scale_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size, scaling_factor);
    return out;
}
"""

swish_scale_cpp_source = """
torch::Tensor swish_scale_cuda(torch::Tensor x, float scaling_factor);
"""

# Compile the inline CUDA code
swish_scale = load_inline(
    name="swish_scale",
    cpp_sources=swish_scale_cpp_source,
    cuda_sources=swish_scale_source,
    functions=["swish_scale_cuda"],
    verbose=True,
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.swish_scale = swish_scale  # The compiled CUDA module

    def forward(self, x):
        x = self.linear(x)
        x = self.swish_scale.swish_scale_cuda(x, self.scaling_factor)
        return x

batch_size = 128
in_features = 32768
out_features = 32768
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]