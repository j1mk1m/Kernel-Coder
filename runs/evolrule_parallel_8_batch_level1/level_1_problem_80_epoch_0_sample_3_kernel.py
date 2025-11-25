import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for convolution
custom_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int output_h, int output_w,
    int input_stride0, int input_stride1, int input_stride2,
    int weight_stride0, int weight_stride1, int weight_stride2,
    int output_stride0, int output_stride1, int output_stride2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_h * output_w) return;

    int n = idx / (out_channels * output_h * output_w);
    int rem = idx % (out_channels * output_h * output_w);
    int c_out = rem / (output_h * output_w);
    int rem2 = rem % (output_h * output_w);
    int h_out = rem2 / output_w;
    int w_out = rem2 % output_w;

    scalar_t sum = 0.0;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_h_val = padding_h + h_out * stride_h + kh * dilation_h;
                int input_w_val = padding_w + w_out * stride_w + kw * dilation_w;

                // Check if input coordinates are valid
                if (input_h_val < 0 || input_h_val >= input_h || input_w_val < 0 || input_w_val >= input_w) {
                    continue;
                }

                // Calculate weight index
                int weight_idx = c_out * weight_stride0 + c_in * weight_stride1 + kh * weight_stride2 + kw;

                // Calculate input index
                int input_idx = n * input_stride0 + c_in * input_stride1 + input_h_val * input_stride2 + input_w_val;

                sum += weight[weight_idx] * input[input_idx];
            }
        }
    }

    // Calculate output index
    int output_idx = n * output_stride0 + c_out * output_stride1 + h_out * output_stride2 + w_out;
    output[output_idx] = sum;
}

torch::Tensor custom_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation
) {
    // Get parameters
    int stride_h = stride;
    int stride_w = stride;

    int padding_h = std::get<0>(padding);
    int padding_w = std::get<1>(padding);

    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    // Input dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);

    // Weight dimensions
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    // Compute output dimensions
    int output_h = (input_h + 2 * padding_h - (kernel_h - 1) * dilation_h - 1) / stride_h + 1;
    int output_w = (input_w + 2 * padding_w - (kernel_w - 1) * dilation_w - 1) / stride_w + 1;

    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, output_h, output_w}, input.options());

    // Strides
    int input_stride0 = input.stride(0); // N
    int input_stride1 = input.stride(1); // C_in
    int input_stride2 = input.stride(2); // H_in

    int weight_stride0 = weight.stride(0); // C_out
    int weight_stride1 = weight.stride(1); // C_in
    int weight_stride2 = weight.stride(2); // K_h

    int output_stride0 = output.stride(0); // C_out
    int output_stride1 = output.stride(1); // H_out
    int output_stride2 = output.stride(2); // W_out

    // Grid and block dimensions
    int total_threads = batch_size * out_channels * output_h * output_w;
    const int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "custom_conv2d_cuda", ([&] {
        custom_conv2d_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_h, input_w,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            output_h, output_w,
            input_stride0, input_stride1, input_stride2,
            weight_stride0, weight_stride1, weight_stride2,
            output_stride0, output_stride1, output_stride2
        );
    }));

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

custom_conv2d_cpp_source = """
torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, std::tuple<int, int> padding, std::tuple<int, int> dilation);
"""

custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=custom_conv2d_cpp_source,
    cuda_sources=custom_conv2d_source,
    functions=["custom_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14", "--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # Initialize weights similarly to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return custom_conv.custom_conv2d_cuda(x, self.weight, self.stride, self.padding, self.dilation)