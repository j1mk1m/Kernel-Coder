import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for ConvTranspose3D
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> output,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int input_depth,
    const int input_height,
    const int input_width) {

    const int batch_idx = blockIdx.x;
    const int out_d = blockIdx.y;
    const int out_h = blockIdx.z * blockDim.y + threadIdx.y;
    const int out_w = threadIdx.x;

    if (out_h >= output_height || out_w >= output_width) return;

    const int col_depth = out_d / kernel_size;
    const int col_height = out_h / kernel_size;
    const int col_width = out_w / kernel_size;
    const int filt_depth = out_d % kernel_size;
    const int filt_height = out_h % kernel_size;
    const int filt_width = out_w % kernel_size;

    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int out_ch = 0; out_ch < out_channels; out_ch++) {
            const int in_d = (col_depth - padding) * stride + filt_depth * dilation;
            const int in_h = (col_height - padding) * stride + filt_height * dilation;
            const int in_w = (col_width - padding) * stride + filt_width * dilation;

            if (in_d >= 0 && in_d < input_depth &&
                in_h >= 0 && in_h < input_height &&
                in_w >= 0 && in_w < input_width) {
                atomicAdd(&output[batch_idx][out_ch][col_depth][col_height][col_width],
                    input[batch_idx][in_ch][in_d][in_h][in_w] * weight[out_ch][in_ch][filt_depth][filt_height][filt_width]);
            }
        }
    }
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);

    const int kernel_size = weight.size(2); // Assuming cube kernel (depth == height == width)
    const int out_channels = weight.size(0);

    // Compute output dimensions using transposed convolution formula
    const int output_depth = (input_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    const int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    const int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const dim3 threads(32, 8); // Threads per block (x, y)
    dim3 blocks(
        batch_size,
        (output_depth * kernel_size), // blockIdx.y covers all possible depth positions
        (output_height + threads.y - 1) / threads.y // blockIdx.z
    );

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
            weight.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
            output.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            output_depth,
            output_height,
            output_width,
            input_depth,
            input_height,
            input_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

# Compile the inline CUDA code
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources="",
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # Initialize weight tensor similar to PyTorch's ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = conv_transpose3d.conv_transpose3d_cuda(
            x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output