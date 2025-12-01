import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise 2D convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int height_in, int width_in, int kernel_size, int stride, int padding) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z * blockDim.z + threadIdx.z;
    int w = blockIdx.w * blockDim.w + threadIdx.w;

    if (b >= batch_size || c >= in_channels || h >= height_in || w >= width_in) {
        return;
    }

    float sum = 0.0f;
    int kh_start = max(0, h - padding);
    int kw_start = max(0, w - padding);
    int kh_end = min(kernel_size, height_in - h + padding);
    int kw_end = min(kernel_size, width_in - w + padding);

    for (int kh = 0; kh < kh_end; ++kh) {
        for (int kw = 0; kw < kw_end; ++kw) {
            int ih = h + kh - padding;
            int iw = w + kw - padding;
            sum += input[b * in_channels * height_in * width_in + c * height_in * width_in + ih * width_in + iw] * weight[c * kernel_size * kernel_size + kh * kernel_size + kw];
        }
    }

    output[b * in_channels * height_in * width_in + c * height_in * width_in + h * width_in + w] = sum;
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, in_channels, (height_in + 2 * padding - kernel_size) / stride + 1, (width_in + 2 * padding - kernel_size) / stride + 1}, input.options());

    dim3 block_size(16, 16, 1);
    dim3 grid_size((width_in + 2 * padding - kernel_size + stride - 1) / stride / block_size.x,
                   (height_in + 2 * padding - kernel_size + stride - 1) / stride / block_size.y,
                   batch_size * in_channels);

    depthwise_conv2d_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, height_in, width_in, kernel_size, stride, padding);

    return output;
}
"""

depthwise_conv2d_cpp_source = (
    "torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding);"
)

# Compile the inline CUDA code for depthwise 2D convolution
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return depthwise_conv2d.depthwise_conv2d_cuda(x, self.weight, self.stride, self.padding)

# Test code
batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
width_in = 512
height_in = 256
stride = 1
padding = 0

def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]