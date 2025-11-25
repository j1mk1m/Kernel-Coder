import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

pointwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void pointwise_conv_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int in_channels,
    int out_channels,
    int H,
    int W
) {
    int block_idx = blockIdx.x;
    int b = block_idx / (H * W);
    int rem = block_idx % (H * W);
    int h = rem / W;
    int w = rem % W;

    int c_out = threadIdx.x;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        int input_offset = b * in_channels * H * W + c_in * H * W + h * W + w;
        sum += input[input_offset] * weight[c_out * in_channels + c_in];
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    int output_offset = b * out_channels * H * W + c_out * H * W + h * W + w;
    output[output_offset] = sum;
}

torch::Tensor pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int B = input.size(0);
    int in_channels = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int out_channels = weight.size(0);

    if (weight.size(1) != in_channels) {
        throw std::runtime_error("Incompatible in_channels and weight dimensions");
    }
    if (bias.defined() && bias.size(0) != out_channels) {
        throw std::runtime_error("Bias size must match out_channels");
    }

    auto output = torch::empty({B, out_channels, H, W}, input.options());

    dim3 threads(out_channels);
    dim3 blocks(B * H * W);

    if (threads.x > 1024) {
        throw std::runtime_error("Number of output channels exceeds maximum threads per block (1024)");
    }

    pointwise_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, in_channels, out_channels, H, W
    );

    return output;
}
"""

pointwise_conv_cpp_source = (
    "torch::Tensor pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

pointwise_conv = load_inline(
    name="pointwise_conv",
    cpp_sources=pointwise_conv_cpp_source,
    cuda_sources=pointwise_conv_source,
    functions=["pointwise_conv_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize weights and bias similar to PyTorch's Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return pointwise_conv.pointwise_conv_cuda(x, self.weight, self.bias)
        else:
            return pointwise_conv.pointwise_conv_cuda(x, self.weight, torch.empty(0, device=x.device, dtype=x.dtype))