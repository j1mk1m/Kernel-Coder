import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for depthwise convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int height, int width,
    int kernel_size, int stride, int padding) {
    
    const int H_out = (height + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (width + 2 * padding - kernel_size) / stride + 1;

    int n = blockIdx.x / in_channels;
    int c = blockIdx.x % in_channels;
    int h_out = threadIdx.y;
    int w_out = threadIdx.x;

    extern __shared__ scalar_t sdata[];

    int th = threadIdx.y;
    int tw = threadIdx.x;

    scalar_t sum = 0.0;

    for (int ph = 0; ph < kernel_size; ph++) {
        for (int pw = 0; pw < kernel_size; pw++) {
            int h = h_out * stride - padding + ph;
            int w = w_out * stride - padding + pw;

            bool valid = (h >= 0 && h < height) && (w >= 0 && w < width);
            scalar_t val = valid ? input[n][c][h][w] : 0;
            sdata[ph * kernel_size + pw] = val;

            __syncthreads();

            sum += sdata[ph * kernel_size + pw] * weight[c][0][ph][pw];
            __syncthreads();
        }
    }

    output[n][c][h_out][w_out] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int H_out = (height + 2 * padding - kernel_size) / stride + 1;
    int W_out = (width + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::zeros({batch_size, in_channels, H_out, W_out}, input.options());

    dim3 threads(kernel_size, kernel_size);
    dim3 blocks(batch_size * in_channels, 1);

    size_t smem_size = kernel_size * kernel_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads, smem_size>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, height, width,
            kernel_size, stride, padding);
    }));

    return output;
}
"""

depthwise_conv_cpp_source = """
#include <torch/extension.h>
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding);
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.bias = None

    def forward(self, x):
        output = depthwise_conv.depthwise_conv2d_cuda(
            x, self.weight, self.kernel_size, self.stride, self.padding
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output