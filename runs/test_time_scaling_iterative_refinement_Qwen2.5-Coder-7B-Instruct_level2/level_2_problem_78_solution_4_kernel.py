import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_kernel(float* input, float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size, int stride, int padding) {
    int b = blockIdx.x / (depth_out * height_out * width_out);
    int d_o = (blockIdx.x % (depth_out * height_out * width_out)) / (height_out * width_out);
    int h_o = ((blockIdx.x % (depth_out * height_out * width_out)) % (height_out * width_out)) / width_out;
    int w_o = ((blockIdx.x % (depth_out * height_out * width_out)) % (height_out * width_out)) % width_out;

    float sum = 0.0f;
    for (int c = 0; c < in_channels; ++c) {
        for (int k_d = 0; k_d < kernel_size; ++k_d) {
            for (int k_h = 0; k_h < kernel_size; ++k_h) {
                for (int k_w = 0; k_w < kernel_size; ++k_w) {
                    int i_d = d_o * stride - padding + k_d;
                    int i_h = h_o * stride - padding + k_h;
                    int i_w = w_o * stride - padding + k_w;
                    if (i_d >= 0 && i_d < depth_in && i_h >= 0 && i_h < height_in && i_w >= 0 && i_w < width_in) {
                        int i_idx = b * in_channels * depth_in * height_in * width_in + c * depth_in * height_in * width_in + i_d * height_in * width_in + i_h * width_in + i_w;
                        int w_idx = b * out_channels * in_channels * kernel_size * kernel_size * kernel_size + c * kernel_size * kernel_size * kernel_size + k_d * kernel_size * kernel_size + k_h * kernel_size + k_w;
                        sum += input[i_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    int o_idx = b * out_channels * depth_out * height_out * width_out + d_o * height_out * width_out + h_o * width_out + w_o;
    output[o_idx] = sum;
}

torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(1);
    int depth_in = input.size(2);
    int height_in = input.size(3);
    int width_in = input.size(4);
    int depth_out = (depth_in - 1) * stride + kernel_size - 2 * padding;
    int height_out = (height_in - 1) * stride + kernel_size - 2 * padding;
    int width_out = (width_in - 1) * stride + kernel_size - 2 * padding;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * depth_out * height_out * width_out + block_size - 1) / block_size;

    conv_transpose_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth_in, height_in, width_in, depth_out, height_out, width_out, kernel_size, stride, padding);

    return output;
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for convolution transpose
conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose

    def forward(self, x):
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size).cuda()  # Example weight tensor
        return self.conv_transpose.conv_transpose_cuda(x, weight)