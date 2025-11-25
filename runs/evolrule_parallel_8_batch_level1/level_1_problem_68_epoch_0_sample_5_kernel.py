import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA implementation for ConvTranspose3d
conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int depth, int width, int height,
    int out_channels, int kernel_depth, int kernel_width, int kernel_height,
    int stride_depth, int stride_width, int stride_height,
    int padding_depth, int padding_width, int padding_height,
    int output_padding_depth, int output_padding_width, int output_padding_height,
    int groups) {

    const int out_depth = output.size(2);
    const int out_width = output.size(3);
    const int out_height = output.size(4);

    const int d = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.w; // Assuming we have a way to handle batch dimension

    if (h >= out_height || w >= out_width) return;

    for (int c_out = 0; c_out < out_channels; c_out += groups) {
        const int c_in_group = c_out / groups;
        const int c_in = c_in_group * groups;

        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                for (int kh = 0; kh < kernel_height; ++kh) {
                    // Compute input coordinates
                    int input_d = (d - kd - padding_depth) / stride_depth + padding_depth;
                    int input_w = (w - kw - padding_width) / stride_width + padding_width;
                    int input_h = (h - kh - padding_height) / stride_height + padding_height;

                    // Check validity of input coordinates
                    if (input_d < 0 || input_d >= depth ||
                        input_w < 0 || input_w >= width ||
                        input_h < 0 || input_h >= height) {
                        continue;
                    }

                    // Compute output coordinates
                    int out_d = d + output_padding_depth;
                    int out_w = w + output_padding_width;
                    int out_h = h + output_padding_height;

                    // Apply the convolution
                    scalar_t val = weight[c_out][c_in_group][kd][kw][kh] * input[b][c_in][input_d][input_w][input_h];
                    atomicAdd(&output[b][c_out][out_d][out_w][out_h], val);
                }
            }
        }
    }
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_depth, int stride_width, int stride_height,
    int padding_depth, int padding_width, int padding_height,
    int output_padding_depth, int output_padding_width, int output_padding_height,
    int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth = input.size(2);
    const int width = input.size(3);
    const int height = input.size(4);

    const int out_channels = weight.size(0);
    const int kernel_depth = weight.size(2);
    const int kernel_width = weight.size(3);
    const int kernel_height = weight.size(4);

    // Compute output dimensions
    const int out_depth = (depth - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth;
    const int out_width = (width - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width;
    const int out_height = (height - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, out_depth, out_width, out_height}, output_options);

    dim3 threads(16, 16); // Block dimensions
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        out_depth
    );

    // Launch kernel
    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
        weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
        output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
        batch_size, in_channels, depth, width, height,
        out_channels, kernel_depth, kernel_width, kernel_height,
        stride_depth, stride_width, stride_height,
        padding_depth, padding_width, padding_height,
        output_padding_depth, output_padding_width, output_padding_height,
        groups
    );

    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, int stride_depth, int stride_width, int stride_height, int padding_depth, int padding_width, int padding_height, int output_padding_depth, int output_padding_width, int output_padding_height, int groups);"
)

# Compile the inline CUDA code for conv_transpose3d
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=["-arch=sm_75"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0),
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight and bias similar to ConvTranspose3d
        kernel_depth, kernel_width, kernel_height = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_depth, kernel_width, kernel_height))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters (simplified, should match PyTorch's initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Attach the CUDA function
        self.conv_transpose3d = conv_transpose3d

    def forward(self, x):
        # Unpack parameters
        stride_d, stride_w, stride_h = self.stride
        padding_d, padding_w, padding_h = self.padding
        output_padding_d, output_padding_w, output_padding_h = self.output_padding

        # Call the custom CUDA kernel
        output = self.conv_transpose3d.conv_transpose3d_cuda(
            x, self.weight,
            stride_d, stride_w, stride_h,
            padding_d, padding_w, padding_h,
            output_padding_d, output_padding_w, output_padding_h,
            self.groups
        )

        # Add bias if needed
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output

# Ensure get_inputs and get_init_inputs remain unchanged
def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]