import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_channels * height * width) {
        int o_channel = idx / (height * width);
        int o_height = (idx % (height * width)) / width;
        int o_width = idx % width;

        float sum = 0.0f;
        for (int c = 0; c < in_channels; ++c) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int i_height = o_height + kh - kernel_size / 2;
                    int i_width = o_width + kw - kernel_size / 2;
                    if (i_height >= 0 && i_height < height && i_width >= 0 && i_width < width) {
                        int i_idx = (c * height + i_height) * width + i_width;
                        int w_idx = (o_channel * in_channels + c) * kernel_size * kernel_size + kh * kernel_size + kw;
                        sum += input[i_idx] * weight[w_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (output.numel() + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size);"
)

# Compile the inline CUDA code for convolution
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
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = convolution

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight, batch_size, in_channels, out_channels, height, width, kernel_size)
        x = torch.nn.functional.hardswish(x)
        x = torch.relu(x)
        return x