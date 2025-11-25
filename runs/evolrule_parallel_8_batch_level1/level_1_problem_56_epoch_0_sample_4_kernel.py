import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

custom_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/tensor_view_io.h>
#include <cutlass/util/reference/host/tensor_fill.h>

template <typename Element>
struct CustomConv2d {
    using Layout = cutlass::layout::TensorNCHW;
    using ElementAccumulator = Element;

    using Operator = cutlass::conv::device::ImplicitGemmConvolution<
        cutlass::conv::Conv2dProblemSize,
        cutlass::TensorRef<Element, Layout>,
        cutlass::TensorRef<Element, Layout>,
        cutlass::TensorRef<Element, Layout>,
        ElementAccumulator,
        Layout,
        cutlass::conv::StridedDilation<cutlass::layout::TensorNCHW>,
        cutlass::conv::StridedDilation<cutlass::layout::TensorNCHW>,
        cutlass::AlgoConstantTiling>;

    static void run(
        cutlass::conv::Conv2dProblemSize problem_size,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor output) {
        Operator conv_op;
        cutlass::Status status = conv_op(
            problem_size,
            output.data_ptr<Element>(),
            input.data_ptr<Element>(),
            weight.data_ptr<Element>(),
            nullptr, // bias
            nullptr);

        if (status != cutlass::Status::kSuccess) {
            AT_ERROR("Cutlass convolution failed");
        }
    }
};

torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups) {
    // Error checks
    assert(groups == 1 && "Groups > 1 not supported");
    assert(dilation_h == 1 && dilation_w == 1 && "Dilation > 1 not supported");

    // Get input dimensions
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Get weight dimensions
    int out_channels = weight.size(0);
    int in_channels_per_group = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    // Calculate output dimensions
    int H_out = (H + 2 * padding_h - kernel_h) / stride_h + 1;
    int W_out = (W + 2 * padding_w - kernel_w) / stride_w + 1;

    // Create output tensor
    auto output = torch::zeros({N, out_channels, H_out, W_out}, input.options());

    // Set up problem size
    cutlass::conv::Conv2dProblemSize problem_size(
        N, C, H, W,
        out_channels, kernel_h, kernel_w,
        H_out, W_out,
        padding_h, padding_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        cutlass::conv::Mode::kCrossCorrelation,
        1 // split_k_slices
    );

    // Launch Cutlass operator
    using Element = float;
    CustomConv2d<Element>::run(problem_size, input, weight, output);

    // Apply bias if present
    if (bias.defined()) {
        auto expanded_bias = bias.view({1, -1, 1, 1});
        output += expanded_bias;
    }

    return output;
}
"""

custom_conv_cpp_source = """
torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups);
"""

# Compile the CUDA code
custom_conv = load_inline(
    name="custom_conv",
    cpp_sources=custom_conv_cpp_source,
    cuda_sources=custom_conv_source,
    functions=["custom_conv2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_present = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize weights similar to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Unpack parameters
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        dilation_h, dilation_w = self.dilation
        groups = self.groups

        # Prepare bias tensor
        if self.bias is not None:
            bias_tensor = self.bias
        else:
            bias_tensor = torch.tensor([], device=x.device)

        # Call the custom CUDA function
        return custom_conv.custom_conv2d_cuda(
            x,
            self.weight,
            bias_tensor,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups
        )