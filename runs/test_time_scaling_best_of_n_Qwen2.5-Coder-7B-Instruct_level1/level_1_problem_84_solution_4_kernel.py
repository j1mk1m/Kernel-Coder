import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for depthwise 2D convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(float* input, float* weight, float* output, int batch_size, int in_channels, int height_in, int width_in, int kernel_size, int stride, int padding) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z;
    int w = blockIdx.w;

    int h_in = h * stride - padding;
    int w_in = w * stride - padding;

    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_out = h_in + kh;
            int w_out = w_in + kw;

            if (h_out >= 0 && h_out < height_in && w_out >= 0 && w_out < width_in) {
                atomicAdd(&output[b * out_channels * height_in * width_in + c * height_in * width_in + h_out * width_in + w_out], 
                           input[b * in_channels * height_in * width_in + c * height_in * width_in + h_in * width_in + w_in] * weight[c * kernel_size * kernel_size + kh * kernel_size + kw]);
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

    auto output = torch::zeros({batch_size, in_channels, height_in, width_in}, input.options());

    const int block_size = 16;
    dim3 grid_size((width_in + block_size - 1) / block_size, (height_in + block_size - 1) / block_size, batch_size, in_channels);
    dim3 block_size(block_size, block_size, 1);

    depthwise_conv2d_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, height_in, width_in, kernel_size, stride, padding);

    return output;
}
"""

depthwise_conv2d_cpp_source = (
    "torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding);"
)

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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return depthwise_conv2d_cuda(x, self.weight, self.stride, self.padding)

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