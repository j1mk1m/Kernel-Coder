import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) return;

    int b = idx / (out_channels * depth * height * width);
    int o = (idx % (out_channels * depth * height * width)) / (depth * height * width);
    int d = ((idx % (out_channels * depth * height * width)) % (depth * height * width)) / (height * width);
    int h = ((idx % (out_channels * depth * height * width)) % (height * width)) / width;
    int w = idx % width;

    float sum = 0.0f;
    for (int c = 0; c < in_channels; ++c) {
        for (int kd = -kernel_size / 2; kd <= kernel_size / 2; ++kd) {
            for (int kh = -kernel_size / 2; kh <= kernel_size / 2; ++kh) {
                for (int kw = -kernel_size / 2; kw <= kernel_size / 2; ++kw) {
                    int icd = d + kd;
                    int ich = h + kh;
                    int icw = w + kw;
                    if (icd >= 0 && icd < depth && ich >= 0 && ich < height && icw >= 0 && icw < width) {
                        int ic_idx = b * in_channels * depth * height * width + c * depth * height * width + icd * height * width + ich * width + icw;
                        int w_idx = o * in_channels * kernel_size * kernel_size * kernel_size + c * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw;
                        sum += input[ic_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    output[idx] = sum;
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, depth, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * depth * height * width + block_size - 1) / block_size;

    conv3d_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_size);

    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D convolution
conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = conv3d
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv.conv3d_cuda(x, self.weight)
        x = x / self.divisor
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x