import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int in_height,
    int in_width,
    int out_height,
    int out_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_height * out_width) return;

    int b = idx / (out_channels * out_height * out_width);
    int residual = idx % (out_channels * out_height * out_width);
    int k = residual / (out_height * out_width);
    residual %= (out_height * out_width);
    int ox = residual / out_width;
    int oy = residual % out_width;

    float acc = 0.0f;

    for (int c = 0; c < in_channels; ++c) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int ix = (ox + padding - i * dilation);
                int iy = (oy + padding - j * dilation);

                if (ix % stride != 0 || iy % stride != 0) continue;

                ix /= stride;
                iy /= stride;

                if (ix < 0 || ix >= in_height || iy < 0 || iy >= in_width) continue;

                int weight_idx = c * out_channels * kernel_size * kernel_size +
                                 k * kernel_size * kernel_size +
                                 i * kernel_size + j;
                float w = weight[weight_idx];

                int input_offset = b * in_channels * in_height * in_width +
                                   c * in_height * in_width +
                                   ix * in_width + iy;
                float in_val = input[input_offset];

                acc += w * in_val;
            }
        }
    }

    if (bias != nullptr) {
        acc += bias[k];
    }

    int output_offset = b * out_channels * out_height * out_width +
                        k * out_height * out_width +
                        ox * out_width + oy;
    output[output_offset] = acc;
}

torch::Tensor conv_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

    int in_height = input.size(2);
    int in_width = input.size(3);

    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto output = torch::empty({input.size(0), out_channels, out_height, out_width},
                              input.options());

    int total_elements = output.numel();

    const int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        input.size(0),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        in_height,
        in_width,
        out_height,
        out_width);

    return output;
}
"""

conv_transpose_cpp_source = """
torch::Tensor conv_transpose_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation);
"""

# Compile the inline CUDA code
conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights similar to PyTorch's default for ConvTranspose2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        return conv_transpose.conv_transpose_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            bias.contiguous() if bias is not None else None,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )