import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* kernel,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K,
    int stride, int padding,
    int H_out, int W_out,
    int has_bias
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out_idx = idx / W_out;
    int h_out = h_out_idx % H_out;
    int c_n = h_out_idx / H_out;
    int c = c_n % C;
    int n = c_n / C;

    float sum = 0.0f;

    for (int k_h = 0; k_h < K; ++k_h) {
        for (int k_w = 0; k_w < K; ++k_w) {
            int h_in = h_out * stride + k_h - padding;
            int w_in = w_out * stride + k_w - padding;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                int input_offset = n * C * H * W + c * H * W + h_in * W + w_in;
                float in_val = input[input_offset];
                int kernel_offset = c * K * K + k_h * K + k_w;
                float ker_val = kernel[kernel_offset];
                sum += in_val * ker_val;
            }
        }
    }

    if (has_bias) {
        sum += bias[c];
    }

    int output_offset = n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out;
    output[output_offset] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor bias,
    int stride,
    int padding,
    bool has_bias
) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = kernel.size(2);

    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    dim3 threads(256);
    int total_threads = N * C * H_out * W_out;
    dim3 blocks((total_threads + threads.x - 1) / threads.x);

    depthwise_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C, H, W,
        K,
        stride, padding,
        H_out, W_out,
        has_bias ? 1 : 0
    );

    return output;
}
"""

depthwise_conv_cpp_source = """
extern "C" {
    torch::Tensor depthwise_conv2d_cuda(
        torch::Tensor input,
        torch::Tensor kernel,
        torch::Tensor bias,
        int stride,
        int padding,
        bool has_bias
    );
}
"""

depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty((out_channels, 1, kernel_size, kernel_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        # Initialize the weights and bias like PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.depthwise_conv = depthwise_conv  # the loaded module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias_tensor = self.bias.contiguous() if self.bias is not None else torch.empty(0, device=x.device)
        return self.depthwise_conv.depthwise_conv2d_cuda(
            x, 
            weight, 
            bias_tensor,
            self.stride, 
            self.padding,
            self.has_bias
        )