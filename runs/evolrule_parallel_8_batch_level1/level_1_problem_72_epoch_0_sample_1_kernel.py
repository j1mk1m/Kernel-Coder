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
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_depth * out_height * out_width) return;

    // Compute output indices
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int d_out = (idx / (out_width * out_height)) % out_depth;
    int c_out = (idx / (out_width * out_height * out_depth)) % out_channels;
    int n = idx / (out_channels * out_depth * out_height * out_width);

    int out_channels_per_group = out_channels / groups;
    int group = c_out / out_channels_per_group;
    int c_out_in_group = c_out % out_channels_per_group;

    int in_channels_per_group = in_channels / groups;

    float acc = 0.0;

    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Compute input indices
                int d_in = (d_out + padding_d - kd + output_padding_d) / stride_d;
                int h_in = (h_out + padding_h - kh + output_padding_h) / stride_h;
                int w_in = (w_out + padding_w - kw + output_padding_w) / stride_w;

                // Check bounds
                if (d_in < 0 || d_in >= in_depth) continue;
                if (h_in < 0 || h_in >= in_height) continue;
                if (w_in < 0 || w_in >= in_width) continue;

                // Iterate over input channels in this group
                for (int c_in_in_group = 0; c_in_in_group < in_channels_per_group; ++c_in_in_group) {
                    int c_in = group * in_channels_per_group + c_in_in_group;

                    // Compute input value index
                    int input_offset = n * in_channels * in_depth * in_height * in_width +
                                      c_in * in_depth * in_height * in_width +
                                      d_in * in_height * in_width +
                                      h_in * in_width + w_in;
                    float input_val = input[input_offset];

                    // Compute weight index
                    int weight_offset = (c_in * out_channels_per_group + c_out_in_group) * kernel_d * kernel_h * kernel_w +
                                       kd * kernel_h * kernel_w +
                                       kh * kernel_w + kw;

                    float weight_val = weight[weight_offset];

                    acc += input_val * weight_val;
                }
            }
        }
    }

    // Write output
    int output_offset = n * out_channels * out_depth * out_height * out_width +
                        c_out * out_depth * out_height * out_width +
                        d_out * out_height * out_width +
                        h_out * out_width + w_out;

    output[output_offset] = acc;
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, 
                                   int kernel_size_d, int kernel_size_h, int kernel_size_w,
                                   int stride_d, int stride_h, int stride_w,
                                   int padding_d, int padding_h, int padding_w,
                                   int output_padding_d, int output_padding_h, int output_padding_w,
                                   int groups) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight.size(1) * groups;

    int out_depth = (in_depth - 1) * stride_d - 2 * padding_d + kernel_size_d + output_padding_d;
    int out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_size_h + output_padding_h;
    int out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_size_w + output_padding_w;

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width},
                              input.options());

    int threads_per_block = 256;
    int total_elements = output.numel();
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose3d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        kernel_size_d, kernel_size_h, kernel_size_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        groups,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width);

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int kernel_size_d, int kernel_size_h, int kernel_size_w,
                                   int stride_d, int stride_h, int stride_w,
                                   int padding_d, int padding_h, int padding_w,
                                   int output_padding_d, int output_padding_h, int output_padding_w,
                                   int groups);
"""

# Load the CUDA extension
conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weight parameter
        self.weight = nn.Parameter(torch.empty(
            (in_channels, out_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2])
        ))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.groups
        )