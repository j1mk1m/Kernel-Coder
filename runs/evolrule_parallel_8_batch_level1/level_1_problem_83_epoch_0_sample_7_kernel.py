import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]

# Custom CUDA kernel for optimized depthwise 2D convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
  
    const int output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = width;

    CUDA_1D_KERNEL_LOOP(index, batch_size * channels * output_height * output_width) {
        int w_out = index % output_width;
        int h_out = (index / output_width) % output_height;
        int c = (index / (output_width * output_height)) % channels;
        int n = index / (channels * output_height * output_width);

        int h_in = h_out * stride - padding;
        scalar_t val = 0;
        for (int k = 0; k < kernel_size; ++k) {
            int h = h_in + dilation * k;
            if (h >= 0 && h < height) {
                val += input[n * channels * height * width + c * height * width + h * width + w_out] *
                    weight[c * kernel_size + k];
            }
        }
        output[index] = val;
    }
}

std::tuple<torch::Tensor> depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int kernel_size = weight.size(1); // weight shape: [channels, kernel_size]

    const int output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = width;

    auto output = torch::zeros({batch_size, channels, output_height, output_width}, input.options());

    const int num_elements = batch_size * channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            channels,
            height,
            width,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv_cpp_source = """
std::tuple<torch::Tensor> depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation);
"""

# Compile the custom CUDA kernel
depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize weights similar to PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(in_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Load the custom CUDA kernel
        self.depthwise_conv = depthwise_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise_conv.depthwise_conv2d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation
        )
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out