import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise division by a scalar
elementwise_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_divide_kernel(const float* x, float scalar, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] / scalar;
    }
}

torch::Tensor elementwise_divide_cuda(torch::Tensor x, float scalar) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_divide_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), scalar, out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_divide_cpp_source = (
    "torch::Tensor elementwise_divide_cuda(torch::Tensor x, float scalar);"
)

# Compile the inline CUDA code for element-wise division
elementwise_divide = load_inline(
    name="elementwise_divide",
    cpp_sources=elementwise_divide_cpp_source,
    cuda_sources=elementwise_divide_source,
    functions=["elementwise_divide_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by
        self.elementwise_divide = elementwise_divide  # Assign the loaded module

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.elementwise_divide.elementwise_divide_cuda(x, self.divide_by)
        return x