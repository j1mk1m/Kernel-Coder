import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom GELU CUDA kernel implementation
gelu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

__global__ void gelu_kernel(const float* __restrict__ x, float* y, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float xi = x[idx];
    const float arg = M_SQRT_2 / M_SQRT_PI * (xi + 0.044715f * xi * xi * xi);
    const float tanh_arg = tanh(arg);
    y[idx] = 0.5f * xi * (1.0f + tanh_arg);
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    const int size = x.numel();
    auto y = torch::empty_like(x);

    const int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gelu_kernel<<<blocks, THREADS_PER_BLOCK>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);
    return y;
}

"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

# Compile the custom CUDA kernel
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu_cuda = gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_cuda.gelu_cuda(x.cuda())  # Ensure input is on GPU