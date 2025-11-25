import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4> input,
    const torch::PackedTensorAccessor<scalar_t,4> weight,
    torch::PackedTensorAccessor<scalar_t,4> output,
    int in_channels, int out_channels, int kernel_h, int kernel_w,
    int stride, int padding_h, int padding_w, int output_padding,
    int groups, int out_h, int out_w) {

    const int B = blockIdx.z;
    const int C = blockIdx.y * blockDim.z + threadIdx.z;
    const int Y = blockIdx.x * blockDim.y + threadIdx.y;
    const int X = threadIdx.x;

    if (C >= out_channels || Y >= out_h || X >= out_w) return;

    // Compute the input position
    int in_y = (Y - padding_h) / stride;
    int in_x = (X - padding_w) / stride;
    if ((Y - padding_h) % stride != 0 || (X - padding_w) % stride != 0) return;

    scalar_t val = 0;
    for (int k_h = 0; k_h < kernel_h; ++k_h) {
        for (int k_w = 0; k_w < kernel_w; ++k_w) {
            const int input_y = in_y - k_h;
            const int input_x = in_x - k_w;
            if (input_y < 0 || input_y >= input.size(2) || input_x < 0 || input_x >= input.size(3)) {
                continue;
            }
            for (int g = 0; g < groups; ++g) {
                int in_c = C / (out_channels / groups) * (in_channels / groups) + g * (in_channels / groups);
                val += weight[C][g * (kernel_h * kernel_w) + k_h * kernel_w + k_w] * 
                       input[B][in_c][input_y][input_x];
            }
        }
    }
    output[B][C][Y][X] = val;
}

std::tuple<int, int> compute_output_size(
    int input_h, int input_w, int kernel_h, int kernel_w, int stride, int padding, int output_padding) {
    int output_h = (input_h - 1) * stride - 2 * padding + kernel_h + output_padding;
    int output_w = (input_w - 1) * stride - 2 * padding + kernel_w + output_padding;
    return {output_h, output_w};
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride, int padding_h, int padding_w, int output_padding, int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_h = input.size(2);
    const int input_w = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    auto output_size = compute_output_size(input_h, input_w, kernel_h, kernel_w, stride, padding_h, output_padding);
    int out_h = output_size.first, out_w = output_size.second;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, input.options());

    const int threads_per_block = 256;
    dim3 threads(32, 8, 1); // X, Y, Z (threadIdx.x, threadIdx.y, threadIdx.z)
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        (out_channels + threads.y * threads.z - 1) / (threads.y * threads.z),
        batch_size
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&]{
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            weight.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            in_channels, out_channels, kernel_h, kernel_w,
            stride, padding_h, padding_w, output_padding, groups, out_h, out_w);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose2d_cpp_source = """
#include <vector>
std::tuple<int, int> compute_output_size(
    int input_h, int input_w, int kernel_h, int kernel_w, int stride, int padding, int output_padding);
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride, int padding_h, int padding_w, int output_padding, int groups);
"""

# Compile the inline CUDA code
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.padding_h, self.padding_w = self.padding
        self.output_padding = output_padding
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
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
        # Ensure tensors are on the same device
        x = x.cuda()
        self.weight = self.weight.cuda()
        if self.bias is not None:
            self.bias = self.bias.cuda()

        output = conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding_h,
            self.padding_w,
            self.output_padding,
            self.groups
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output