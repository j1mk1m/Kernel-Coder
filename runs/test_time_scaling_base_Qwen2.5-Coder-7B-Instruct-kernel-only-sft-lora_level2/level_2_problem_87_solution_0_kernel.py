import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Mish activation
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mish_kernel(const float* input, float* output, int elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        output[idx] = input[idx] * tanh(log(1 + exp(input[idx])));
    }
}

torch::Tensor mish_cuda(torch::Tensor input) {
    auto elements = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (elements + block_size - 1) / block_size;

    mish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), elements);

    return output;
}
"""

mish_cpp_source = (
    "torch::Tensor mish_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Mish activation
mish = load_inline(
    name="mish",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_source,
    functions=["mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.subtraction = subtraction
        self.elementwise_add = elementwise_add

    def forward(self, x):
        x = self.conv(x)
        x = self.elementwise_add.elementwise_add_cuda(x, torch.tensor([self.subtract_value_1], device=x.device))
        x = self.elementwise_add.elementwise_add_cuda(x, torch.tensor([self.subtract_value_2], device=x.device))
        x = mish.mish_cuda(x)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]