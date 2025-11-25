import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int groups,
    int out_depth,
    int out_height,
    int out_width,
    bool has_bias,
    const float* bias) {

    int out_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_index >= batch_size * out_channels * out_depth * out_height * out_width) {
        return;
    }

    // Unflatten the output index into coordinates
    int n = out_index / (out_channels * out_depth * out_height * out_width);
    int rem = out_index % (out_channels * out_depth * out_height * out_width);

    int f_out = rem / (out_depth * out_height * out_width);
    rem %= (out_depth * out_height * out_width);

    int d_out = rem / (out_height * out_width);
    rem %= (out_height * out_width);

    int h_out = rem / out_width;
    int w_out = rem % out_width;

    float acc = 0.0;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    // Compute input coordinates
                    int d_in = (d_out - kd + 2 * padding - output_padding) / stride;
                    int h_in = (h_out - kh + 2 * padding - output_padding) / stride;
                    int w_in = (w_out - kw + 2 * padding - output_padding) / stride;

                    // Check boundaries
                    if (d_in < 0 || d_in >= in_depth) continue;
                    if (h_in < 0 || h_in >= in_height) continue;
                    if (w_in < 0 || w_in >= in_width) continue;

                    // Weight index calculation
                    const int kernel_offset = kd * kernel_size * kernel_size +
                                             kh * kernel_size +
                                             kw;
                    const int weight_offset = c_in * out_channels * kernel_size * kernel_size * kernel_size +
                                             f_out * kernel_size * kernel_size * kernel_size +
                                             kernel_offset;

                    // Input and weight access
                    const float in_val = input[n * in_channels * in_depth * in_height * in_width +
                                               c_in * in_depth * in_height * in_width +
                                               d_in * in_height * in_width +
                                               h_in * in_width +
                                               w_in];
                    
                    const float w_val = weight[weight_offset];
                    acc += in_val * w_val;
                }
            }
        }
    }

    if (has_bias) {
        acc += bias[f_out];
    }

    // Output index calculation
    const int output_offset = n * out_channels * out_depth * out_height * out_width +
                             f_out * out_depth * out_height * out_width +
                             d_out * out_height * out_width +
                             h_out * out_width +
                             w_out;
    output[output_offset] = acc;
}

torch::Tensor conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int kernel_size = weight.size(3); // Assuming square kernel
    const int out_channels = weight.size(1) * groups; // For groups=1, this simplifies to weight.size(1)

    // Output dimensions calculation
    const int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    const int threads_per_block = 256;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        out_depth,
        out_height,
        out_width,
        bias.defined(),
        bias.defined() ? bias.data_ptr<float>() : nullptr
    );

    return output;
}
"""

conv_transpose3d_cpp_source = (
    "torch::Tensor conv_transpose3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int output_padding, int dilation, int groups);"
)

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_forward"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=["--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, output_padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize parameters with the same shape as PyTorch's ConvTranspose3d
        self.weight = nn.Parameter(torch.empty(
            (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters similar to PyTorch's default initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return conv_transpose3d.conv_transpose3d_forward(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups
        )