import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Custom CUDA kernel for optimized asymmetric convolution
convolution_kernel = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void asymmetric_conv2d_kernel(
    const torch::PackedTensorAccessor<scalar_t,4> input,
    const torch::PackedTensorAccessor<scalar_t,4> weight,
    const torch::PackedTensorAccessor<scalar_t,1> bias,
    torch::PackedTensorAccessor<scalar_t,4> output,
    const int in_channels,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const bool has_bias) {

    const int batch_size = output.size(0);
    const int out_h = output.size(2);
    const int out_w = output.size(3);
    const int channels_per_group = in_channels / groups;

    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * out_channels * out_h * out_w) {
        return;
    }

    int w = output_idx % out_w;
    int h = (output_idx / out_w) % out_h;
    int c_out = (output_idx / (out_w * out_h)) % out_channels;
    int n = output_idx / (out_channels * out_w * out_h);

    int group = c_out / (out_channels / groups);
    c_out = c_out % (out_channels / groups);

    scalar_t val = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = -padding_h + h * stride + kh * dilation_h;
            int w_in = -padding_w + w * stride + kw * dilation_w;
            if (h_in >= 0 && h_in < input.size(2) && w_in >=0 && w_in < input.size(3)) {
                for (int c_in = 0; c_in < channels_per_group; ++c_in) {
                    val += weight[group * (out_channels/groups) * kernel_h * kernel_w * channels_per_group 
                                + c_out * kernel_h * kernel_w * channels_per_group 
                                + kh * kernel_w * channels_per_group 
                                + kw * channels_per_group 
                                + c_in] 
                            * input[n][group * channels_per_group + c_in][h_in][w_in];
                }
            }
        }
    }

    if (has_bias) {
        val += bias[c_out + group * (out_channels/groups)];
    }

    output[n][group * (out_channels/groups) + c_out][h][w] = val;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asymmetric_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups) {

    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int batch_size = input.size(0);
    const int input_h = input.size(2);
    const int input_w = input.size(3);

    int out_h = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

    const bool has_bias = bias.has_value();

    int threads = 256;
    int elements = batch_size * out_channels * out_h * out_w;
    int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "asymmetric_conv2d", ([&] {
        using scalar_t = scalar_t;

        auto input_acc = input.packed_accessor<scalar_t,4>();
        auto weight_acc = weight.packed_accessor<scalar_t,4>();
        auto output_acc = output.packed_accessor<scalar_t,4>();

        auto bias_acc = has_bias ? bias.value().packed_accessor<scalar_t,1>() : 
                                  torch::Tensor().packed_accessor<scalar_t,1>();

        asymmetric_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            input_acc,
            weight_acc,
            has_bias ? bias_acc : bias_acc,
            output_acc,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            groups,
            has_bias
        );
    }));

    return std::make_tuple(output, input, weight, bias.value_or(torch::Tensor()));
}

// CPU fallback (not implemented, just placeholder)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asymmetric_conv2d_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups) {
    AT_ERROR("CPU version not implemented");
}
"""

# Compile the CUDA extension
asymmetric_conv = load_inline(
    name="asymmetric_conv",
    cpp_sources="""
#include <torch/extension.h>
""",
    cuda_sources=convolution_kernel,
    functions=[
        "std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asymmetric_conv2d(torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, int, int, int, int, int, int)",
        "std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> asymmetric_conv2d_cpu(torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, int, int, int, int, int, int)"
    ],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding_h = padding if isinstance(padding, int) else padding[0]
        self.padding_w = padding if isinstance(padding, int) else padding[1]
        self.dilation_h = dilation if isinstance(dilation, int) else dilation[0]
        self.dilation_w = dilation if isinstance(dilation, int) else dilation[1]
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias like PyTorch's Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        bias_opt = self.bias if self.bias is not None else torch.tensor([])
        output, _, _, _ = asymmetric_conv.asymmetric_conv2d(
            x,
            self.weight,
            bias_opt,
            self.stride,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
            self.groups
        )
        return output