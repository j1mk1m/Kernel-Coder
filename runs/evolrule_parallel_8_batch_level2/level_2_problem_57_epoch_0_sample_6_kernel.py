import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused Conv2D + ReLU + HardSwish CUDA kernel
fused_conv_relu_hswish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_conv_relu_hswish_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size) {

    const int kernel_radius = kernel_size / 2;
    const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_row = blockIdx.y;
    const int output_channel = blockIdx.z;

    if (output_col >= width || output_row >= height) return;

    scalar_t sum = (bias ? bias[output_channel] : 0.0);

    for (int i = 0; i < in_channels; ++i) {
        for (int kh = -kernel_radius; kh <= kernel_radius; ++kh) {
            for (int kw = -kernel_radius; kw <= kernel_radius; ++kw) {
                int h = output_row + kh;
                int w = output_col + kw;
                if (h >= 0 && h < height && w >= 0 && w < width) {
                    int input_offset = i * height * width + h * width + w;
                    int weight_offset = output_channel * in_channels * kernel_size * kernel_size +
                        i * kernel_size * kernel_size +
                        (kh + kernel_radius) * kernel_size +
                        (kw + kernel_radius);
                    sum += input[input_offset] * weight[weight_offset];
                }
            }
        }
    }

    // Apply ReLU
    sum = fmax(sum, 0.0);

    // Apply HardSwish
    scalar_t scale = (sum + 3.0) * (1.0 / 6.0);
    scale = fmax(scale, 0.0);
    scale = fmin(scale, 1.0);
    sum *= scale;

    // Write to output
    int output_offset = output_channel * height * width + output_row * width + output_col;
    output[output_offset] = sum;
}

torch::Tensor fused_conv_relu_hswish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size) {

    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = weight.size(0);
    const auto height = input.size(2);
    const auto width = input.size(3);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    dim3 threads(256);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        height,
        out_channels
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv_relu_hswish_cuda", ([&] {
        fused_conv_relu_hswish_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            (bias.defined() ? bias.data_ptr<scalar_t>() : nullptr),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_size
        );
    }));

    return output;
}
"""

# Compile the fused kernel
fused_conv_relu_hswish = load_inline(
    name="fused_conv_relu_hswish",
    cuda_sources=fused_conv_relu_hswish_source,
    functions=["fused_conv_relu_hswish_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        return fused_conv_relu_hswish.fused_conv_relu_hswish_cuda(
            x.contiguous(),
            self.weight,
            self.bias,
            self.kernel_size
        )

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]