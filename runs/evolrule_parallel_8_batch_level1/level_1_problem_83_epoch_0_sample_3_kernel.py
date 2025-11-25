import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <bool HAS_BIAS>
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_height,
    int output_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_channels * output_height * output_width) {
        return;
    }

    // Compute output indices
    int w_out = idx % output_width;
    int rem = idx / output_width;
    int h_out = rem % output_height;
    rem /= output_height;
    int c = rem % in_channels;
    rem /= in_channels;
    int n = rem;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = h_out * stride + kh * dilation - padding;
        if (h_in < 0 || h_in >= input_height) continue;

        // Input offset calculation
        const int input_offset = 
            n * in_channels * input_height * input_width +
            c * input_height * input_width +
            h_in * input_width +
            w_out;

        const float in_val = input[input_offset];

        // Weight offset calculation
        const int weight_offset = c * kernel_size + kh;
        const float w_val = weight[weight_offset];

        sum += in_val * w_val;
    }

    if (HAS_BIAS) {
        sum += bias[c];
    }

    // Output offset calculation
    const int output_offset = 
        n * in_channels * output_height * output_width +
        c * output_height * output_width +
        h_out * output_width +
        w_out;

    output[output_offset] = sum;
}

void depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_height,
    int output_width,
    bool has_bias
) {
    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;

    if (has_bias) {
        depthwise_conv2d_kernel<true><<<blocks, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation,
            output_height,
            output_width
        );
    } else {
        depthwise_conv2d_kernel<false><<<blocks, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            nullptr,
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation,
            output_height,
            output_width
        );
    }
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    // Compute dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int kernel_size = weight.size(2); // since weight is (C,1,K,1)

    // Calculate effective kernel height for output height
    int effective_kernel_height = dilation * (kernel_size - 1) + 1;
    int output_height = (input_height + 2 * padding - effective_kernel_height) / stride + 1;
    int output_width = (input_width + 2 * padding - 1) / stride + 1;

    torch::Tensor output = torch::empty({batch_size, in_channels, output_height, output_width},
        torch::device(input.device()).dtype(input.dtype()));

    depthwise_conv2d_cuda(
        input,
        weight,
        has_bias ? bias : torch::Tensor(),
        output,
        batch_size,
        in_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        output_height,
        output_width,
        has_bias
    );

    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
);
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_forward"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        has_bias = self.bias is not None
        bias = self.bias if has_bias else None

        # Call the CUDA function
        return depthwise_conv.depthwise_conv2d_forward(
            x,
            self.weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            has_bias
        )

def get_inputs():
    # The original get_inputs is kept as is
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]