import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for depthwise-separable convolution
fused_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_conv2d(
    const scalar_t* input,
    const scalar_t* depthwise_weight,
    const scalar_t* depthwise_bias,
    const scalar_t* pointwise_weight,
    const scalar_t* pointwise_bias,
    scalar_t* output,
    int batch, int in_channels, int out_channels,
    int height, int width, int kernel_h, int kernel_w,
    int stride, int padding, int dilation,
    int height_out, int width_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_channels * height_out * width_out) return;

    int w_out = idx % width_out;
    int rem = idx / width_out;
    int h_out = rem % height_out;
    rem /= height_out;
    int o = rem % out_channels;
    int b = rem / out_channels;

    scalar_t acc = 0;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        scalar_t depthwise_val = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride + (kh - padding) * dilation;
                int w_in = w_out * stride + (kw - padding) * dilation;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_offset = 
                        b * in_channels * height * width +
                        c_in * height * width +
                        h_in * width + w_in;
                    scalar_t input_val = input[input_offset];

                    int dw_weight_offset = 
                        c_in * kernel_h * kernel_w +
                        kh * kernel_w + kw;
                    scalar_t dw_weight = depthwise_weight[dw_weight_offset];

                    depthwise_val += input_val * dw_weight;
                }
            }
        }
        if (depthwise_bias) {
            depthwise_val += depthwise_bias[c_in];
        }

        int pw_weight_offset = o * in_channels + c_in;
        acc += depthwise_val * pointwise_weight[pw_weight_offset];
    }

    if (pointwise_bias) {
        acc += pointwise_bias[o];
    }

    int output_offset = 
        b * out_channels * height_out * width_out +
        o * height_out * width_out +
        h_out * width_out + w_out;
    output[output_offset] = acc;
}

std::tuple<torch::Tensor> fused_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor depthwise_bias,
    torch::Tensor pointwise_weight,
    torch::Tensor pointwise_bias,
    int stride, int padding, int dilation,
    int height_out, int width_out
) {
    const auto batch = input.size(0);
    const auto in_channels = input.size(1);
    const auto out_channels = pointwise_weight.size(0);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto kernel_h = depthwise_weight.size(2);
    const auto kernel_w = depthwise_weight.size(3);

    auto output = torch::zeros({batch, out_channels, height_out, width_out}, input.options());

    const int threads_per_block = 256;
    const int num_elements = batch * out_channels * height_out * width_out;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv2d", ([&] {
        fused_conv2d<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
            pointwise_weight.data_ptr<scalar_t>(),
            pointwise_bias.defined() ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch, in_channels, out_channels,
            height, width, kernel_h, kernel_w,
            stride, padding, dilation,
            height_out, width_out
        );
    }));

    cudaDeviceSynchronize();  // Ensure kernel completion
    return output;
}
"""

fused_conv2d_cpp_source = """
torch::Tensor fused_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor depthwise_bias,
    torch::Tensor pointwise_weight,
    torch::Tensor pointwise_bias,
    int stride, int padding, int dilation,
    int height_out, int width_out
);
"""

# Compile the fused CUDA kernel
fused_conv2d = load_inline(
    name="fused_conv2d",
    cpp_sources=[fused_conv2d_cpp_source],
    cuda_sources=[fused_conv2d_source],
    functions=["fused_conv2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise_weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        self.depthwise_bias = nn.Parameter(torch.empty(in_channels)) if bias else None
        self.pointwise_weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        self.pointwise_bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Initialize parameters similarly to PyTorch's default initialization
        nn.init.kaiming_uniform_(self.depthwise_weight, a=math.sqrt(5))
        if self.depthwise_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.depthwise_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.depthwise_bias, -bound, bound)
        nn.init.kaiming_uniform_(self.pointwise_weight, a=math.sqrt(5))
        if self.pointwise_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.pointwise_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.pointwise_bias, -bound, bound)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        N, C, H, W = x.shape
        kernel_h = self.kernel_size
        kernel_w = self.kernel_size
        padding_h = self.padding
        padding_w = self.padding
        dilation_h = self.dilation
        dilation_w = self.dilation
        stride_h = self.stride
        stride_w = self.stride

        effective_kernel_h = dilation_h * (kernel_h - 1) + 1
        effective_kernel_w = dilation_w * (kernel_w - 1) + 1

        height_out = (H + 2 * padding_h - effective_kernel_h) // stride_h + 1
        width_out = (W + 2 * padding_w - effective_kernel_w) // stride_w + 1

        # Prepare bias tensors
        dw_bias = self.depthwise_bias if self.depthwise_bias is not None else torch.Tensor()
        pw_bias = self.pointwise_bias if self.pointwise_bias is not None else torch.Tensor()

        # Call fused kernel
        output = fused_conv2d.fused_conv2d_cuda(
            x,
            self.depthwise_weight,
            dw_bias,
            self.pointwise_weight,
            pw_bias,
            self.stride,
            self.padding,
            self.dilation,
            height_out,
            width_out
        )

        return output