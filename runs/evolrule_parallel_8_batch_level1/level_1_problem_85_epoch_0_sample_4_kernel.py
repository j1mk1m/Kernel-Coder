import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_forward(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int output_height,
    int output_width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * in_channels * output_height * output_width) return;

    int w_out = index % output_width;
    int h_out = (index / output_width) % output_height;
    int c = (index / (output_width * output_height)) % in_channels;
    int n = index / (output_width * output_height * in_channels);

    float sum = 0.0f;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            int w_in = w_out * stride_w - padding_w + kw * dilation_w;

            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                int input_offset = n * in_channels * input_height * input_width +
                                   c * input_height * input_width +
                                   h_in * input_width + w_in;
                int weight_offset = c * kernel_h * kernel_w + kh * kernel_w + kw;

                sum += input[input_offset] * weight[weight_offset];
            }
        }
    }

    int output_offset = n * in_channels * output_height * output_width +
                        c * output_height * output_width +
                        h_out * output_width + w_out;
    output[output_offset] = sum;
}

torch::Tensor depthwise_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int output_h = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_w = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({N, C, output_h, output_w}, input.options());

    int num_elements = N * C * output_h * output_w;
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    depthwise_conv2d_forward<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        output_h, output_w
    );

    cudaDeviceSynchronize();

    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
);
"""

depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels  # Matches original Conv2d setup
        self.kernel_size = (kernel_size_h, kernel_size_w)
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)
        self.groups = groups
        self.bias = bias

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size_h, kernel_size_w))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))  # Output channels = in_channels
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = depthwise_conv2d.depthwise_conv2d_forward_cuda(
            x,
            self.weight,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1]
        )

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out