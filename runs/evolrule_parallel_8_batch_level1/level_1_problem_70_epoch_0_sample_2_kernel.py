import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_forward_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int depth_in,
    int height_in,
    int width_in,
    int depth_out,
    int height_out,
    int width_out,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth_out * height_out * width_out)
        return;

    // Compute output indices
    int b = idx / (out_channels * depth_out * height_out * width_out);
    int c_out = (idx % (out_channels * depth_out * height_out * width_out))
                / (depth_out * height_out * width_out);
    int d_out = (idx % (depth_out * height_out * width_out))
                / (height_out * width_out);
    int h_out = (idx % (height_out * width_out)) / width_out;
    int w_out = idx % width_out;

    float acc = 0.0;

    // Iterate over kernel elements and input channels
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Compute input indices
                int d_in = (d_out + padding - kd - output_padding) / stride;
                int h_in = (h_out + padding - kh - output_padding) / stride;
                int w_in = (w_out + padding - kw - output_padding) / stride;

                // Check bounds
                if (d_in < 0 || d_in >= depth_in || h_in < 0 || h_in >= height_in || w_in < 0 || w_in >= width_in)
                    continue;

                // Iterate over input channels considering groups
                for (int in_c = 0; in_c < in_channels; ++in_c) {
                    // Group handling
                    int group = in_c / (in_channels / groups);
                    int out_c_per_group = out_channels / groups;
                    int w_out_c = c_out % out_c_per_group;
                    int in_c_per_group = in_c % (in_channels / groups);

                    // Weight index calculation
                    int w_offset = (in_c) * (out_channels / groups) * kernel_size * kernel_size * kernel_size
                                  + (w_out_c) * kernel_size * kernel_size * kernel_size
                                  + kd * kernel_size * kernel_size
                                  + kh * kernel_size
                                  + kw;

                    float w_val = weight[w_offset];

                    // Input index calculation
                    int in_offset = b * in_channels * depth_in * height_in * width_in
                                   + in_c * depth_in * height_in * width_in
                                   + d_in * height_in * width_in
                                   + h_in * width_in
                                   + w_in;

                    acc += input[in_offset] * w_val;
                }
            }
        }
    }

    // Write output
    int out_offset = b * out_channels * depth_out * height_out * width_out
                    + c_out * depth_out * height_out * width_out
                    + d_out * height_out * width_out
                    + h_out * width_out
                    + w_out;

    output[out_offset] = acc;
}

torch::Tensor conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int depth_in,
    int height_in,
    int width_in,
    int depth_out,
    int height_out,
    int width_out,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int groups) {

    auto output = torch::empty(
        {batch_size, out_channels, depth_out, height_out, width_out},
        input.options()
    );

    const int threads = 256;
    const int blocks = (output.numel() + threads - 1) / threads;

    conv_transpose3d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, kernel_size,
        depth_in, height_in, width_in,
        depth_out, height_out, width_out,
        stride, padding, output_padding, dilation, groups
    );

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int depth_in,
    int height_in,
    int width_in,
    int depth_out,
    int height_out,
    int width_out,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int groups
);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=conv_transpose3d_cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, 
                 output_padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(
            in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depth_in, height_in, width_in = x.shape[2], x.shape[3], x.shape[4]
        depth_out = (depth_in - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        height_out = (height_in - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        width_out = (width_in - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        output = conv_transpose3d.conv_transpose3d_forward(
            x.contiguous(),
            self.weight.contiguous(),
            x.size(0),
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            depth_in, height_in, width_in,
            depth_out, height_out, width_out,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups
        )

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)

        return output