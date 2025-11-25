import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_3d_kernel(const float* input, const float* weight, float* output, int in_channels, int out_channels, int depth, int height, int width, int kernel_size, int stride, int padding) {
    int idx_out = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_in = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_d = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx_out >= out_channels || idx_in >= in_channels || idx_d >= depth) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int d_in = idx_d * stride - padding + k;
                int h_in = idx_in * stride - padding + i;
                int w_in = idx_out * stride - padding + j;
                if (d_in >= 0 && d_in < depth && h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int idx_input = ((d_in * height + h_in) * width + w_in) * in_channels + idx_in;
                    int idx_weight = ((k * kernel_size + i) * kernel_size + j) * in_channels + idx_in;
                    sum += input[idx_input] * weight[idx_weight];
                }
            }
        }
    }
    output[(idx_d * height + idx_in) * width + idx_out] = sum;
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto out_channels = weight.size(0);
    auto in_channels = weight.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto kernel_size = weight.size(2);
    auto stride = 2; // Assuming fixed stride for simplicity
    auto padding = 1; // Assuming fixed padding for simplicity

    auto output = torch::zeros({out_channels, depth, height, width}, input.options());

    const int block_size = 256;
    const int num_blocks_out = (out_channels + block_size - 1) / block_size;
    const int num_blocks_in = (in_channels + block_size - 1) / block_size;
    const int num_blocks_depth = (depth + block_size - 1) / block_size;

    dim3 grid(num_blocks_out, num_blocks_in, num_blocks_depth);
    dim3 block(block_size, block_size, block_size);

    conv_transpose_3d_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), in_channels, out_channels, depth, height, width, kernel_size, stride, padding);

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


# Define the custom CUDA kernel for batch normalization
batch_norm_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_norm_3d_kernel(const float* input, float* mean, float* var, float* output, float eps, int depth, int height, int width, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= channels * depth * height * width) {
        return;
    }

    int ch = idx / (depth * height * width);
    int d = (idx % (depth * height * width)) / (height * width);
    int h = (idx % (height * width)) / width;
    int w = idx % width;

    float inv_var = 1.0f / sqrt(var[ch]);
    output[idx] = (input[idx] - mean[ch]) * inv_var + eps;
}

torch::Tensor batch_norm_3d_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var) {
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto eps = 1e-5f; // Assuming fixed epsilon for simplicity

    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (channels * depth * height * width + block_size - 1) / block_size;

    dim3 grid((num_blocks + block_size - 1) / block_size);
    dim3 block(block_size);

    batch_norm_3d_kernel<<<grid, block>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), output.data_ptr<float>(), eps, depth, height, width, channels);

    return output;
}
"""

batch_norm_3d_cpp_source = (
    "torch::Tensor batch_norm_3d_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var);"
)

# Compile the inline CUDA code for batch normalization
batch_norm_3d = load_inline(
    name="batch_norm_3d",
    cpp_sources=batch_norm_3d_cpp_source,
    cuda_sources=batch_norm_3d_source,
    functions=["batch_norm_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose_3d
        self.batch_norm = batch_norm_3d

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_3d_cuda(x, torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        x = self.batch_norm.batch_norm_3d_cuda(x, torch.randn(out_channels), torch.randn(out_channels))
        x = F.avg_pool3d(x, kernel_size=2)
        x = F.avg_pool3d(x, kernel_size=2)
        return x


# Example usage
if __name__ == "__main__":
    batch_size = 64
    in_channels = 3
    out_channels = 16
    depth, height, width = 32, 32, 32
    kernel_size = 3
    stride = 2
    padding = 1
    bias_shape = (out_channels, 1, 1, 1)

    model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, bias_shape)
    inputs = torch.randn(batch_size, in_channels, depth, height, width).cuda()

    outputs = model_new(inputs)
    print(outputs.shape)