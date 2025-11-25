import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for pointwise 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_channels * height * width) {
        int b = idx / (out_channels * height * width);
        int o = (idx % (out_channels * height * width)) / (height * width);
        int h = (idx % (out_channels * height * width)) % height;
        int w = idx % width;
        float sum = 0.0f;
        for (int c = 0; c < in_channels; ++c) {
            sum += input[b * in_channels * height * width + c * height * width + h * width + w] * weight[o * in_channels + c];
        }
        output[idx] = sum;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width + block_size - 1) / block_size;

    conv2d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width);

    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for pointwise 2D convolution
conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv2d.conv2d_cuda(x, self.weight)

# Test code
batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]