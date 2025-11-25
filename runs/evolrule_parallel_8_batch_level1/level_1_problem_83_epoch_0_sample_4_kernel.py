import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int H, int W,
    int kernel_size, int stride, int padding, int dilation,
    const scalar_t* __restrict__ bias,
    int H_out, int W_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * in_channels * H_out * W_out) return;

    int n = tid / (in_channels * H_out * W_out);
    int rem = tid % (in_channels * H_out * W_out);
    int c = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int h_out = rem / W_out;
    int w_out = rem % W_out;

    scalar_t sum = 0;

    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = h_out * stride + kh * dilation - padding;
        int w_in = w_out * stride + 0 * dilation - padding;

        if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) continue;

        int input_offset = n * in_channels * H * W
                         + c * H * W
                         + h_in * W + w_in;
        scalar_t in_val = input[input_offset];

        int weight_offset = c * kernel_size + kh;
        scalar_t weight_val = weights[weight_offset];

        sum += in_val * weight_val;
    }

    if (bias) {
        sum += bias[c];
    }

    int output_offset = n * in_channels * H_out * W_out
                      + c * H_out * W_out
                      + h_out * W_out + w_out;
    output[output_offset] = sum;
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int kernel_size
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int H_out = (H + 2 * padding - dilation * (kernel_size -1) -1) / stride + 1;
    int W_out = (W + 2 * padding - dilation * (1 - 1) -1) / stride + 1;

    auto output = torch::empty({batch_size, in_channels, H_out, W_out}, input.options());

    int threads_per_block = 256;
    int elements = batch_size * in_channels * H_out * W_out;
    int blocks_per_grid = (elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "depthwise_conv2d_forward", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, H, W,
            kernel_size, stride, padding, dilation,
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            H_out, W_out
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return depthwise_conv.depthwise_conv2d_forward(
            x,
            self.weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.kernel_size
        )