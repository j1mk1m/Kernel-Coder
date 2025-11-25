import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size, int stride, int padding) {
    int b = blockIdx.z;
    int o = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (o >= out_channels || i >= in_channels || b >= batch_size) {
        return;
    }

    float sum = 0.0f;
    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int id = b * in_channels * depth * height * width + i * depth * height * width + d * height * width + h * width + w;
                int od = b * out_channels * (depth + 2 * padding) * (height + 2 * padding) * (width + 2 * padding) + o * (depth + 2 * padding) * (height + 2 * padding) * (width + 2 * padding) + (d + padding) * (height + 2 * padding) * (width + 2 * padding) + (h + padding) * (width + 2 * padding) + (w + padding);
                sum += input[id] * weight[o * in_channels * kernel_size * kernel_size * kernel_size + i * kernel_size * kernel_size * kernel_size + (d + padding) * kernel_size * kernel_size + (h + padding) * kernel_size + (w + padding)];
            }
        }
    }

    output[od] = sum;
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int kernel_size = weight.size(3);
    int stride = 2; // Assuming fixed stride for simplicity
    int padding = 1; // Assuming fixed padding for simplicity

    auto output = torch::zeros({batch_size, out_channels, depth + 2 * padding, height + 2 * padding, width + 2 * padding}, input.options());

    const int block_size = 32;
    dim3 grid_size((in_channels + block_size - 1) / block_size, (out_channels + block_size - 1) / block_size, batch_size);
    dim3 block_size(block_size, block_size, 1);

    conv_transpose_3d_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_size, stride, padding);

    return output;
}
"""

conv_transpose_3d_cpp_source = (
    "torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose_3d
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_3d_cuda(x, self.weight)
        x = x * self.scale1
        x = self.avg_pool(x)
        x = x + self.bias
        x = x * self.scale2
        return x