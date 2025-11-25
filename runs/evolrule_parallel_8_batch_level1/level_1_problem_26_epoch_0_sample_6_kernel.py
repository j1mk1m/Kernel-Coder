import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom GELU CUDA kernel implementation
gelu_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

__device__ __forceinline__ float fast_gelu(const float x) {
    // Using the approximation for GELU: 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x * x * x)))
    // This is faster than the exact implementation and maintains good accuracy
    float term = sqrt(2.f / static_cast<float>(M_PI)) * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.f + tanh(term));
}

__global__ void gelu_forward_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fast_gelu(input[idx]);
    }
}

torch::Tensor gelu_forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gelu_forward_kernel<<<blocks, THREADS_PER_BLOCK>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

# Compile the CUDA kernel inline
gelu_cpp_source = "torch::Tensor gelu_forward_cuda(torch::Tensor input);"
gelu_extension = load_inline(
    name="gelu_ext",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_kernel_source,
    functions=["gelu_forward_cuda"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_forward = gelu_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_forward.gelu_forward_cuda(x)