import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for optimized ConvTranspose2d
conv_transpose_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int kernel_size, int stride, int padding, int dilation,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width, int output_height, int output_width) {

    const int n = blockIdx.x;
    const int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_w = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_h >= output_height || out_w >= output_width) return;

    scalar_t val = 0;
    for (int k_in = 0; k_in < in_channels; ++k_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Compute effective kernel position (dilation handled)
                const int eff_kh = kh * dilation;
                const int eff_kw = kw * dilation;

                // Compute corresponding input position
                const int in_h = out_h - eff_kh + padding;
                const int in_w = out_w - eff_kw + padding;

                // Check if input position is valid
                if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                    // Transposed convolution uses flipped kernel
                    // So kernel index is (kernel_size-1 - kh, kernel_size-1 - kw)
                    const int kernel_h = kernel_size - 1 - kh;
                    const int kernel_w = kernel_size - 1 - kw;

                    // Accumulate contribution from this kernel weight
                    val += weight[k_in][kernel_h][kernel_w] * input[n][k_in][in_h][in_w];
                }
            }
        }
    }

    output[n][0][out_h][out_w] = val; // Assuming out_channels=1 for simplicity?
}

// Forward function wrapper
torch::Tensor conv_transpose_2d_cuda(torch::Tensor input, torch::Tensor weight,
    int kernel_size, int stride, int padding, int dilation,
    int output_height, int output_width) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    const auto out_channels = weight.size(0); // Assuming weight is [out_channels, in_channels, k, k]

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    dim3 threads(32, 8); // Tune thread block size
    dim3 blocks(batch_size, (output_height + threads.y - 1)/threads.y, (output_width + threads.x -1)/threads.x);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            kernel_size, stride, padding, dilation,
            batch_size, in_channels, out_channels,
            input_height, input_width, output_height, output_width);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

# Header for compilation
conv_transpose_2d_header = (
    "torch::Tensor conv_transpose_2d_cuda(torch::Tensor input, torch::Tensor weight, "
    "int kernel_size, int stride, int padding, int dilation, "
    "int output_height, int output_width);"
)

# Compile the CUDA extension
conv_transpose_2d = load_inline(
    name="conv_transpose_2d",
    cpp_sources=conv_transpose_2d_header,
    cuda_sources=conv_transpose_2d_source,
    functions=["conv_transpose_2d_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=["--expt-extended-lambda"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate output dimensions manually
        batch_size, _, H_in, W_in = x.shape
        H_out = (H_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        W_out = (W_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1

        output = conv_transpose_2d.conv_transpose_2d_cuda(
            x.contiguous(), self.weight.contiguous(),
            self.kernel_size, self.stride, self.padding, self.dilation,
            H_out, W_out
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output