import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size) {
    int b = blockIdx.x / (height * width);
    int h_start = blockIdx.x % (height * width) / width;
    int w_start = blockIdx.x % (height * width) % width;
    int c_out = blockIdx.y;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int d = 0; d < kernel_size; ++d) {
            for (int dh = 0; dh < kernel_size; ++dh) {
                for (int dw = 0; dw < kernel_size; ++dw) {
                    int d_in = d + h_start;
                    int h_in = dh + w_start;
                    int w_in = dw + b;
                    if (d_in >= 0 && d_in < depth && h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = (b * in_channels + c_in) * depth * height * width + (d_in * height + h_in) * width + w_in;
                        int weight_idx = (c_out * in_channels + c_in) * kernel_size * kernel_size * kernel_size + (d * kernel_size * kernel_size + dh * kernel_size + dw);
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    int output_idx = (b * out_channels + c_out) * depth * height * width + (h_start * width + w_start);
    output[output_idx] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, depth, height, width});

    const int block_size = 256;
    const int num_blocks = (batch_size * height * width + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operators for 3D convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.conv.weight)
        x = x * self.scaling_factor
        x = torch.tanh(x)
        x = x * self.bias
        x = torch.sigmoid(x)
        return x


def get_inputs():
    batch_size = 128
    in_channels = 3
    out_channels = 16
    depth, height, width = 16, 64, 64
    kernel_size = 3
    scaling_factor = 2
    bias_shape = (out_channels, 1, 1, 1)

    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]