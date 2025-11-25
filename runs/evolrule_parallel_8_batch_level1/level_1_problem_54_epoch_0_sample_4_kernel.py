import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the CUDA kernel code for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void conv3d_forward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int kernel_size,
    int stride, int padding,
    int out_depth, int out_height, int out_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_depth * out_height * out_width)
        return;

    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int d_out = (idx / (out_width * out_height)) % out_depth;
    int c_out = (idx / (out_width * out_height * out_depth)) % out_channels;
    int n = idx / (out_channels * out_depth * out_height * out_width);

    T sum = static_cast<T>(0.0);
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int id = d_out * stride + kd - padding;
                    int ih = h_out * stride + kh - padding;
                    int iw = w_out * stride + kw - padding;

                    if (id < 0 || id >= in_depth ||
                        ih < 0 || ih >= in_height ||
                        iw < 0 || iw >= in_width) {
                        continue;
                    }

                    int input_offset = n * in_channels * in_depth * in_height * in_width +
                                      c_in * in_depth * in_height * in_width +
                                      id * in_height * in_width +
                                      ih * in_width +
                                      iw;

                    int weight_offset = c_out * in_channels * kernel_size * kernel_size * kernel_size +
                                       c_in * kernel_size * kernel_size * kernel_size +
                                       kd * kernel_size * kernel_size +
                                       kh * kernel_size +
                                       kw;

                    sum += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[idx] = sum;
}

extern "C" {
    torch::Tensor conv3d_forward_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int stride,
        int padding,
        int kernel_size) {

        int batch_size = input.size(0);
        int in_channels = input.size(1);
        int in_depth = input.size(2);
        int in_height = input.size(3);
        int in_width = input.size(4);

        int out_channels = weight.size(0);
        int out_depth = (in_depth + 2 * padding - (kernel_size - 1) - 1) / stride + 1;
        int out_height = (in_height + 2 * padding - (kernel_size - 1) - 1) / stride + 1;
        int out_width = (in_width + 2 * padding - (kernel_size - 1) - 1) / stride + 1;

        torch::Tensor output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

        const float* bias_ptr = nullptr;
        if (bias.defined()) {
            bias_ptr = bias.data_ptr<float>();
        }

        int num_elements = batch_size * out_channels * out_depth * out_height * out_width;
        int block_size = 256;
        int grid_size = (num_elements + block_size - 1) / block_size;

        conv3d_forward_kernel<float><<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            batch_size, in_channels, in_depth, in_height, in_width,
            out_channels, kernel_size,
            stride, padding,
            out_depth, out_height, out_width);

        cudaDeviceSynchronize();
        return output;
    }
}
"""

# Define the C++ header for the function
conv3d_cpp_source = """
torch::Tensor conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int kernel_size);
"""

# Compile the inline CUDA code
conv3d_module = load_inline(
    name="conv3d_module",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-gencode=arch=compute_70,code=sm_70"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups  # Currently not handled in the kernel
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.bias is not None:
            bias_tensor = self.bias
        else:
            bias_tensor = torch.empty(0, device=x.device, dtype=x.dtype)

        output = conv3d_module.conv3d_forward_cuda(
            x,
            self.weight,
            bias_tensor,
            self.stride,
            self.padding,
            self.kernel_size
        )
        return output