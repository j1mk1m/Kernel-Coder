import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D convolution
custom_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void conv1d_forward_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int length, int out_channels, int kernel_size) {
    int b = blockIdx.x / (out_channels * length);
    int o = (blockIdx.x % (out_channels * length)) / length;
    int l = blockIdx.x % length;

    float sum = 0.0f;
    for (int c = 0; c < in_channels; ++c) {
        int i_start = max(l - kernel_size + 1, 0);
        int i_end = min(l + 1, length);
        for (int i = i_start; i < i_end; ++i) {
            sum += input[b * in_channels * length + c * length + i] * weight[o * in_channels * kernel_size + c * kernel_size + l - i];
        }
    }

    if (l >= kernel_size - 1) {
        atomicAdd(&output[b * out_channels * length + o * length + l], sum);
    }
}

torch::Tensor custom_conv1d_forward_cuda(torch::Tensor input, torch::Tensor weight) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, length}, input.options());

    dim3 threads(TILE_WIDTH);
    dim3 blocks((batch_size * out_channels * length + TILE_WIDTH - 1) / TILE_WIDTH);

    conv1d_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, length, out_channels, kernel_size);

    return output;
}
"""

custom_conv_cpp_source = (
    "torch::Tensor custom_conv1d_forward_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 1D convolution
custom_conv = load_inline(
    name="custom_convolution",
    cpp_sources=custom_conv_cpp_source,
    cuda_sources=custom_conv_source,
    functions=["custom_conv1d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        if self.bias is not None:
            return custom_conv.custom_conv1d_forward_cuda(x, self.weight) + self.bias.view(-1, 1, 1)
        else:
            return custom_conv.custom_conv1d_forward_cuda(x, self.weight)


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return [64, 128, 3]