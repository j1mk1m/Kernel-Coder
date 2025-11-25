import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel source
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" __global__ void depthwise_conv2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels,
    int input_height, int input_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int output_height, int output_width,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_channels * output_height * output_width)
        return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c = (idx / (output_width * output_height)) % in_channels;
    int n = idx / (in_channels * output_height * output_width);

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            int w_in = w_out * stride_w - padding_w + kw * dilation_w;

            if (h_in >= 0 && h_in < input_height &&
                w_in >= 0 && w_in < input_width) {
                int input_offset = n * in_channels * input_height * input_width +
                                   c * input_height * input_width +
                                   h_in * input_width + w_in;
                float input_val = input[input_offset];

                int weight_offset = c * kernel_h * kernel_w +
                                    kh * kernel_w + kw;
                float weight_val = weight[weight_offset];

                sum += input_val * weight_val;
            }
        }
    }

    if (has_bias) {
        sum += bias[c];
    }

    int output_offset = n * in_channels * output_height * output_width +
                        c * output_height * output_width +
                        h_out * output_width + w_out;

    output[output_offset] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    bool has_bias
) {
    input = input.contiguous();
    weight = weight.contiguous();
    if (has_bias)
        bias = bias.contiguous();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, in_channels, output_height, output_width}, input.options());

    int total_threads = batch_size * in_channels * output_height * output_width;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    depthwise_conv2d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels,
        input_height, input_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        output_height, output_width,
        has_bias
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));

    return output;
}
"""

# Compile the CUDA code
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        assert out_channels == in_channels, "Depthwise requires out_channels == in_channels"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size_h, kernel_size_w))
        if self.bias:
            self.bias_param = nn.Parameter(torch.empty(in_channels))
        else:
            self.bias_param = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias_param if self.bias else None
        return depthwise_conv2d.depthwise_conv2d_cuda(
            x,
            self.weight,
            bias,
            self.kernel_size_h,
            self.kernel_size_w,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
            self.bias
        )