import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 1D convolution
conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_1d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int length, int kernel_size, int stride, int padding, int dilation) {
    int b = blockIdx.y;
    int o = blockIdx.z;
    int l_o = blockIdx.x * blockDim.x + threadIdx.x;

    if (l_o >= length) return;

    float sum = 0.0f;
    int w_start = (l_o * stride - padding) / dilation;
    int w_end = min(w_start + kernel_size, length);

    for (int i = 0; i < w_end - w_start; ++i) {
        int w_idx = w_start + i;
        int i_idx = (w_idx * dilation - padding + kernel_size - 1) / stride;
        if (i_idx >= 0 && i_idx < length) {
            sum += input[b * in_channels * length + i_idx * in_channels + o] * weight[o * kernel_size + i];
        }
    }

    output[b * out_channels * length + l_o * out_channels + o] = sum;
}

torch::Tensor conv_transpose_1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto length = input.size(2);
    auto kernel_size = weight.size(1);

    auto output = torch::zeros({batch_size, out_channels, length}, input.options());

    dim3 block_size(256, 1, 1);
    dim3 grid_size((length + block_size.x - 1) / block_size.x, batch_size, out_channels);

    conv_transpose_1d_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, length, kernel_size, stride, padding, dilation);

    return output;
}
"""

conv_transpose_1d_cpp_source = (
    "torch::Tensor conv_transpose_1d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"
)

# Compile the inline CUDA code for transposed 1D convolution
conv_transpose_1d = load_inline(
    name="conv_transpose_1d",
    cpp_sources=conv_transpose_1d_cpp_source,
    cuda_sources=conv_transpose_1d_source,
    functions=["conv_transpose_1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv1d_transpose = conv_transpose_1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d_transpose.conv_transpose_1d_cuda(x, self.weight, self.stride, self.padding, self.dilation)

# Initialize the weights manually
model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, dilation)
model_new.weight = nn.Parameter(torch.randn(out_channels, kernel_size))

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
length = 131072
stride = 2
padding = 1
dilation = 2

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]