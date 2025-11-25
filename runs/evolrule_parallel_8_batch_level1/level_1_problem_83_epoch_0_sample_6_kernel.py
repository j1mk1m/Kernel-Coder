import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from torch import Tensor

# Custom CUDA kernel for optimized depthwise convolution
custom_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_height,
    int output_width) {

    const int HW = height * width;
    const int KH = kernel_size;
    const int KW = 1;
    const int dilation_h = dilation;
    const int dilation_w = dilation;
    const int effective_kernel_h = (kernel_size - 1) * dilation_h + 1;
    const int effective_pad_h = padding;
    const int effective_pad_w = padding;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ scalar_t cache[256];

    if (index < batch_size * in_channels * output_height * output_width) {
        int ow = index % output_width;
        int oh = (index / output_width) % output_height;
        int c = (index / (output_width * output_height)) % in_channels;
        int n = index / (in_channels * output_height * output_width);

        scalar_t sum = 0;
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh * stride + kh * dilation_h - effective_pad_h;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow * stride + kw * dilation_w - effective_pad_w;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = n * in_channels * HW + c * width + iw + ih * width;
                    int weight_idx = c * kernel_size + kh;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        output[index] = sum;
    }
}

at::Tensor depthwise_conv2d_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (width + 2 * padding - dilation * (1 - 1) - 1) / stride + 1;

    auto output = at::empty({batch_size, in_channels, output_height, output_width}, input.options());

    int total_threads = batch_size * in_channels * output_height * output_width;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            height,
            width,
            kernel_size,
            stride,
            padding,
            dilation,
            output_height,
            output_width);
    }));

    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depthwise_conv2d_cuda, "Depthwise 2D convolution forward (CUDA)");
}
"""

# Compile the CUDA extension
custom_conv2d = load(
    name="custom_conv2d",
    sources=[custom_conv2d_source],
    extra_cflags=['-gencode=arch=compute_80,code=sm_80', '-O3'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Reshape weight to match kernel_size (kernel_size, 1)
        weight = self.weight.view(self.in_channels, self.kernel_size)
        output = custom_conv2d.forward(
            x,
            weight,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output