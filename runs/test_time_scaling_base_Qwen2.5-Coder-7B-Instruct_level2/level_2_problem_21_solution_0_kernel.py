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
    if (idx >= batch_size * out_channels * height * width) {
        return;
    }

    int o_idx = idx / (height * width);
    int c_idx = (idx % (height * width)) / (height * kernel_size);
    int h_idx = ((idx % (height * width)) % (height * kernel_size)) / kernel_size;
    int w_idx = ((idx % (height * width)) % (height * kernel_size)) % kernel_size;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int ih_idx = h_idx + i;
            int iw_idx = w_idx + j;
            if (ih_idx >= height || iw_idx >= width) {
                continue;
            }
            int in_idx = o_idx * in_channels * height * width + c_idx * height * width + ih_idx * width + iw_idx;
            int wi_idx = c_idx * kernel_size * kernel_size + i * kernel_size + j;
            sum += input[in_idx] * weight[wi_idx];
        }
    }
    output[idx] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height * width + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight);"
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
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv.conversion_cuda(x, self.conv.weight)
        x = x + self.bias
        x = x * self.scale
        x = torch.sigmoid(x)
        x = self.group_norm(x)
        return x