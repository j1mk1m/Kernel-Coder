import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for vectorized ReLU using float4 to process 4 elements per thread
relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vectorized_relu_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = idx * 4;
    if (offset < n) {
        float4 val = ((float4*)in)[offset];
        val.x = fmaxf(val.x, 0.f);
        val.y = fmaxf(val.y, 0.f);
        val.z = fmaxf(val.z, 0.f);
        val.w = fmaxf(val.w, 0.f);
        ((float4*)out)[offset] = val;
    }
}

torch::Tensor vectorized_relu_cuda(torch::Tensor in) {
    auto out = torch::empty_like(in);
    int n = in.numel();
    // Ensure n is divisible by 4 (true for given input dimensions)
    assert(n % 4 == 0);
    const int threads_per_block = 256;
    const int blocks = (n / 4 + threads_per_block - 1) / threads_per_block;
    vectorized_relu_kernel<<<blocks, threads_per_block>>>(out.data_ptr<float>(), in.data_ptr<float>(), n);
    return out;
}
"""

# C++ header for the CUDA function
relu_cpp_source = """
torch::Tensor vectorized_relu_cuda(torch::Tensor in);
"""

# Compile the CUDA extension
vectorized_relu = load_inline(
    name='vectorized_relu',
    cpp_sources=relu_cpp_source,
    cuda_sources=relu_source,
    functions=['vectorized_relu_cuda'],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.relu = vectorized_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu.vectorized_relu_cuda(x)