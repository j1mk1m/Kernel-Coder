import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for depthwise 2D convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int height_in, int width_in, int kernel_size, int stride, int padding) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h_out = blockIdx.z;
    int w_out = blockIdx.w;

    int h_in_start = h_out * stride - padding;
    int w_in_start = w_out * stride - padding;

    for (int h_in = 0; h_in < kernel_size; ++h_in) {
        for (int w_in = 0; w_in < kernel_size; ++w_in) {
            int h_in_idx = h_in_start + h_in;
            int w_in_idx = w_in_start + w_in;

            if (h_in_idx >= 0 && h_in_idx < height_in && w_in_idx >= 0 && w_in_idx < width_in) {
                int in_idx = b * in_channels * height_in * width_in + c * height_in * width_in + h_in_idx * width_in + w_in_idx;
                int weight_idx = c * kernel_size * kernel_size + h_in * kernel_size + w_in;
                int out_idx = b * in_channels * height_out * width_out + c * height_out * width_out + h_out * width_out + w_out;

                atomicAdd(&output[out_idx], input[in_idx] * weight[weight_idx]);
            }
        }
    }
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto kernel_size = weight.size(2);
    auto height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    auto width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, in_channels, height_out, width_out}, input.options());

    const int block_size_h = 16;
    const int block_size_w = 16;
    const int grid_size_b = batch_size;
    const int grid_size_c = in_channels;
    const int grid_size_h_out = height_out;
    const int grid_size_w_out = width_out;

    depthwise_conv2d_kernel<<<grid_size_b, grid_size_c, 0, 0>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, height_in, width_in, kernel_size, stride, padding);

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
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.elementwise_add = elementwise_add
        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return self.depthwise_conv2d.depthwise_conv2d_cuda(x, x, stride, padding)