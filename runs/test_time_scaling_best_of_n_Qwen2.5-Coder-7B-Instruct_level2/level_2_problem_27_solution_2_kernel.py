import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void conv3d_forward_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_depth, int kernel_height, int kernel_width) {
    int b = blockIdx.z;
    int c_out = blockIdx.y;
    int c_in = blockIdx.x;
    int d_out = blockIdx.w;
    int h_out = blockIdx.v;
    int w_out = blockIdx.u;

    float sum = 0.0f;
    for (int k_d = 0; k_d < kernel_depth; ++k_d) {
        for (int k_h = 0; k_h < kernel_height; ++k_h) {
            for (int k_w = 0; k_w < kernel_width; ++k_w) {
                int d_in = d_out + k_d - kernel_depth / 2;
                int h_in = h_out + k_h - kernel_height / 2;
                int w_in = w_out + k_w - kernel_width / 2;

                if (d_in >= 0 && d_in < depth && h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int i = b * in_channels * depth * height * width + c_in * depth * height * width + d_in * height * width + h_in * width + w_in;
                    int j = c_out * in_channels * kernel_depth * kernel_height * kernel_width + c_in * kernel_depth * kernel_height * kernel_width + k_d * kernel_height * kernel_width + k_h * kernel_width + k_w;
                    sum += input[i] * weight[j];
                }
            }
        }
    }

    int o_idx = b * out_channels + c_out;
    atomicAdd(&output[o_idx], sum);
}

torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto kernel_depth = weight.size(2);
    auto kernel_height = weight.size(3);
    auto kernel_width = weight.size(4);

    auto output = torch::zeros({batch_size, out_channels}, input.options());

    dim3 blocks_per_grid(batch_size, out_channels, in_channels, (depth + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

    conv3d_forward_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_depth, kernel_height, kernel_width);

    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for convolution
conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for mean pooling
mean_pooling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_pooling_forward_kernel(const float* input, float* output, int batch_size, int in_channels, int depth, int height, int width, int pool_depth, int pool_height, int pool_width) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int d_out = blockIdx.x;
    int h_out = blockIdx.w;
    int w_out = blockIdx.v;

    float sum = 0.0f;
    int count = 0;
    for (int k_d = 0; k_d < pool_depth; ++k_d) {
        for (int k_h = 0; k_h < pool_height; ++k_h) {
            for (int k_w = 0; k_w < pool_width; ++k_w) {
                int d_in = d_out * pool_depth + k_d;
                int h_in = h_out * pool_height + k_h;
                int w_in = w_out * pool_width + k_w;

                if (d_in >= 0 && d_in < depth && h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int i = b * in_channels * depth * height * width + c * depth * height * width + d_in * height * width + h_in * width + w_in;
                    sum += input[i];
                    count++;
                }
            }
        }
    }

    int o_idx = b * in_channels + c;
    atomicAdd(&output[o_idx], sum / count);
}

torch::Tensor mean_pooling_forward_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto pool_depth = 2;
    auto pool_height = 2;
    auto pool_width = 2;

    auto output = torch::zeros({batch_size, in_channels}, input.options());

    dim3 blocks_per_grid(batch_size, in_channels, (depth + pool_depth - 1) / pool_depth, (height + pool_height - 1) / pool_height, (width + pool_width - 1) / pool_width);
    dim3 threads_per_block(1, 1, 1);

    mean_pooling_forward_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, depth, height, width, pool_depth, pool_height, pool_width);

    return output;
}
"""

mean_pooling_cpp_source = (
    "torch::Tensor mean_pooling_forward_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for mean pooling
mean_pooling = load_inline(
    name="mean_pooling",
    cpp_sources=mean_pooling_cpp_source,
    cuda_sources=mean_pooling_source,
    functions=["mean_pooling_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = conv3d
        self.mean_pooling = mean_pooling

    def forward(self, x):
        x = self.conv.conv3d_forward_cuda(x, self.weight)     # (B, C, D, H, W)
        x = F.hardswish(x)                                   # Nonlinear activation
        x = self.group_norm(x)                               # Normalization over channels
        x = self.mean_pooling.mean_pooling_forward_cuda(x)   # Mean over spatial dims → (B, C)
        return x

    def weight(self):
        return torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]).cuda()

# === Test config ===
batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = (4, 4, 4)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

# Example usage
model_new = ModelNew(in_channels, out_channels, kernel_size)
inputs = get_inputs()
outputs = model_new(inputs[0])
print(outputs.shape)  # Should print torch.Size([1024, 16])