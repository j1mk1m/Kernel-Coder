import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
transposed_convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel function for 3D transposed convolution
__global__ void transposed_convolution_kernel(float* input, float* weight, float* output, int input_depth, int input_height, int input_width, int output_depth, int output_height, int output_width, int channels, int groups) {
    int out_idx = blockIdx.y * output_width + blockIdx.x;
    int group_idx = blockIdx.z;
    int ch_idx = group_idx * channels;

    if (out_idx >= output_width * output_height * output_depth || ch_idx >= channels) {
        return;
    }

    float sum = 0.0f;
    int stride_d = output_width * output_height;
    int stride_h = output_width;
    int stride_w = 1;
    int input_idx = 0;
    int weight_idx = 0;

    for (int d = 0; d < output_depth; ++d) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                int out_offset = d * stride_d + h * stride_h + w * stride_w;
                for (int c = 0; c < channels; ++c) {
                    input_idx = (ch_idx + c) * input_depth * input_height * input_width;
                    weight_idx = (group_idx * channels + c) * output_depth * output_height * output_width;
                    sum += input[input_idx + out_offset] * weight[weight_idx + out_offset];
                }
            }
        }
    }

    output[out_idx] = sum;
}

torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, int output_depth, int output_height, int output_width, int channels, int groups) {
    auto out = torch::zeros({output_depth, output_height, output_width, channels, groups}, input.options());

    const int block_size = 256;
    const int num_blocks = (output_width * output_height * output_depth + block_size - 1) / block_size;

    transposed_convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), input.size(2), input.size(3), input.size(4), output_depth, output_height, output_width, channels, groups);

    return out;
}
"""

transposed_convolution_cpp_source = (
    "torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, int output_depth, int output_height, int output_width, int channels, int groups);"
)

# Compile the inline CUDA code for 3D transposed convolution
transposed_convolution = load_inline(
    name="transposed_convolution",
    cpp_sources=transposed_convolution_cpp_source,
    cuda_sources=transposed_convolution_source,
    functions=["transposed_convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.transposed_convolution = transposed_convolution
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.transposed_convolution.transposed_convolution_cuda(x, self.weight, self.output_depth, self.output_height, self.output_width, self.channels, self.groups)
        x = x * self.scale_factor
        x = self.batch_norm(x)
        x = self.global_avg_pool(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 16, 32, 32
kernel_size = 5
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]