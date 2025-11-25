import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise subtraction
elementwise_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_subtract_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

torch::Tensor elementwise_subtract_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_subtract_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_subtract_cpp_source = (
    "torch::Tensor elementwise_subtract_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise subtraction
elementwise_subtract = load_inline(
    name="elementwise_subtract",
    cpp_sources=elementwise_subtract_cpp_source,
    cuda_sources=elementwise_subtract_source,
    functions=["elementwise_subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.elementwise_subtract = elementwise_subtract

    def forward(self, x):
        x = self.conv(x)
        x = self.elementwise_subtract.elementwise_subtract_cuda(x, torch.tensor([self.subtract_value], device=x.device))
        x = torch.nn.functional.hardswish(x)
        x = self.pool(x)
        x = torch.nn.functional.mish(x)
        return x