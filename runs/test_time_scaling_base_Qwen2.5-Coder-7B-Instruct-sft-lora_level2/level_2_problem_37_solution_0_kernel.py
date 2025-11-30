import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused matrix multiplication and Swish activation
fused_matmul_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_matmul_swish_kernel(const float* a, const float* b, float* out, int batch_size, int in_features, int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            sum += a[row * in_features + i] * b[i * out_features + col];
        }
        out[row * out_features + col] = sum * sigmoid(sum);
    }
}

torch::Tensor fused_matmul_swish_cuda(torch::Tensor a, torch::Tensor b) {
    auto batch_size = a.size(0);
    auto in_features = a.size(1);
    auto out_features = b.size(1);
    auto out = torch::zeros({batch_size, out_features}, a.options());

    const int block_size = 256;
    const int num_blocks_x = (out_features + block_size - 1) / block_size;
    const int num_blocks_y = (batch_size + block_size - 1) / block_size;

    fused_matmul_swish_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features, out_features);

    return out;
}
"""

fused_matmul_swish_cpp_source = (
    "torch::Tensor fused_matmul_swish_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for fused matrix multiplication and Swish activation
fused_matmul_swish = load_inline(
    name="fused_matmul_swish",
    cpp_sources=fused_matmul_swish_cpp_source,
    cuda_sources=fused_matmul_swish_source,
    functions=["fused_matmul_swish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for GroupNorm
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_kernel(const float* x, const float* mean, const float* inv_std, float* out, int batch_size, int in_features, int num_groups) {
    int group = blockIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.z * blockDim.z + threadIdx.z;

    if (group < num_groups && row < batch_size && col < in_features) {
        int group_size = in_features / num_groups;
        int group_offset = group * group_size;
        int group_col = col % group_size;
        int group_row = row;

        float sum = 0.0f;
        for (int i = 0; i < group_size; ++i) {
            sum += x[group_row * in_features + group_offset + i];
        }
        float mean_val = sum / group_size;

        float var_sum = 0.0f;
        for (int i = 0; i < group_size; ++i) {
            var_sum += pow(x[group_row * in_features + group_offset + i] - mean_val, 2);
        }
        float var_val = var_sum / group_size;

        float norm_val = (x[group_row * in_features + group_offset + group_col] - mean_val) * inv_std[group_col];
        out[group_row * in_features + group_offset + group_col] = norm_val;
    }
}

torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor inv_std) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto num_groups = mean.size(0);
    auto out = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks_x = num_groups;
    const int num_blocks_y = (batch_size + block_size - 1) / block_size;
    const int num_blocks_z = (in_features + block_size - 1) / block_size;

    group_norm_kernel<<<dim3(num_blocks_x, num_blocks_y, num_blocks_z), dim3(block_size, block_size, block_size)>>>(x.data_ptr<float>(), mean.data_ptr<float>(), inv_std.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features, num_groups);

    return out;
}
"""

group_norm_cpp_source = (
    "torch::Tensor group_norm_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor inv_std);"
)

# Compile the inline CUDA code for GroupNorm
group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.fused_matmul_swish = fused_matmul_swish
        self.group_norm = group_norm

    def forward(self, x):
        x = self.fused_matmul_swish.fused_matmul_swish_cuda(x, self.weight)
        x = self.bias + x
        x = self.group_norm.group_norm_cuda(x, self.mean, self.inv_std)
        return x

    def init_weights(self, weight, bias, mean, inv_std):
        self.weight = weight
        self.bias = bias
        self.mean = mean
        self.inv_std = inv_std