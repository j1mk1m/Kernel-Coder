import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 3D convolution
transposed_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_conv_kernel(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels, int depth_in, int width_in, int height_in,
    int depth_out, int width_out, int height_out, int kernel_depth, int kernel_width, int kernel_height,
    int stride_depth, int stride_width, int stride_height, int padding_depth, int padding_width, int padding_height,
    int dilation_depth, int dilation_width, int dilation_height) {

    int n = blockIdx.x; // batch index
    int c_out = blockIdx.y; // output channel index
    int d_out = blockIdx.z / (width_out * height_out);
    int w_out = (blockIdx.z % (width_out * height_out)) / height_out;
    int h_out = blockIdx.z % height_out;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k_d = 0; k_d < kernel_depth; ++k_d) {
            for (int k_w = 0; k_w < kernel_width; ++k_w) {
                for (int k_h = 0; k_h < kernel_height; ++k_h) {
                    int d_in = d_out * stride_depth - padding_depth + k_d * dilation_depth;
                    int w_in = w_out * stride_width - padding_width + k_w * dilation_width;
                    int h_in = h_out * stride_height - padding_height + k_h * dilation_height;
                    if (d_in >= 0 && d_in < depth_in && w_in >= 0 && w_in < width_in && h_in >= 0 && h_in < height_in) {
                        int i = n * in_channels * depth_in * width_in * height_in +
                                c_in * depth_in * width_in * height_in +
                                d_in * width_in * height_in +
                                w_in * height_in +
                                h_in;
                        int j = n * out_channels * depth_out * width_out * height_out +
                                c_out * depth_out * width_out * height_out +
                                d_out * width_out * height_out +
                                w_out * height_out +
                                h_out;
                        int k = c_in * kernel_depth * kernel_width * kernel_height +
                                k_d * kernel_width * kernel_height +
                                k_w * kernel_height +
                                k_h;
                        sum += input[i] * weight[j * kernel_depth * kernel_width * kernel_height + k];
                    }
                }
            }
        }
    }

    int o = n * out_channels * depth_out * width_out * height_out +
            c_out * depth_out * width_out * height_out +
            d_out * width_out * height_out +
            w_out * height_out +
            h_out;
    output[o] = sum;
}

torch::Tensor transposed_conv_cuda(
    torch::Tensor input, torch::Tensor weight, int stride_depth, int stride_width, int stride_height,
    int padding_depth, int padding_width, int padding_height, int dilation_depth, int dilation_width, int dilation_height) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth_in = input.size(2);
    auto width_in = input.size(3);
    auto height_in = input.size(4);
    auto depth_out = (depth_in - 1) * stride_depth + 2 * padding_depth - (kernel_depth - 1) * dilation_depth + 1;
    auto width_out = (width_in - 1) * stride_width + 2 * padding_width - (kernel_width - 1) * dilation_width + 1;
    auto height_out = (height_in - 1) * stride_height + 2 * padding_height - (kernel_height - 1) * dilation_height + 1;

    auto output = torch::zeros({batch_size, out_channels, depth_out, width_out, height_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * depth_out * width_out * height_out + block_size - 1) / block_size;

    transposed_conv_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, depth_in, width_in, height_in,
        depth_out, width_out, height_out, kernel_depth, kernel_width, kernel_height,
        stride_depth, stride_width, stride_height, padding_depth, padding_width, padding_height,
        dilation_depth, dilation_width, dilation_height);

    return output;
}
"""

transposed_conv_cpp_source = (
    "torch::Tensor transposed_conv_cuda(torch::Tensor input, torch::Tensor weight, int stride_depth, int stride_width, int stride_height, int padding_depth, int padding_width, int padding_height, int dilation_depth, int dilation_width, int dilation_height);"
)

# Compile the inline CUDA code for transposed 3D convolution
transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources=transposed_conv_cpp_source,
    cuda_sources=transposed_conv_source,
    functions=["transposed_conv_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride_depth, stride_width, stride_height = self.stride
        padding_depth, padding_width, padding_height = self.padding
        dilation_depth, dilation_width, dilation_height = 1, 1, 1  # Assuming no dilation for now

        return transposed_conv.transposed_conv_cuda(
            x, self.weight, stride_depth, stride_width, stride_height,
            padding_depth, padding_width, padding_height, dilation_depth, dilation_width, dilation_height
        )


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, width, height).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]  # Provide in_channels, out_channels, kernel_size for initialization