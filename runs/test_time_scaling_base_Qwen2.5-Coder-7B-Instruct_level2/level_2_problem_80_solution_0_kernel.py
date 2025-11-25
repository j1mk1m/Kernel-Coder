import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM and subsequent operations
gemm_max_sub_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_max_sub_gelu_kernel(const float* x, const float* weight, float* out, int batch_size, int in_features, int out_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= out_features) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        sum += x[row * in_features + i] * weight[i * out_features + col];
    }

    out[row * out_features + col] = sum;
    if (col == 0) {
        out[row * out_features + col] = torch::max(out[row * out_features + col], 0);
        out[row * out_features + col] -= torch::mean(out[row * out_features : (row + 1) * out_features]);
        out[row * out_features + col] = torch::gelu(out[row * out_features + col]);
    }
}

torch::Tensor gemm_max_sub_gelu_cuda(torch::Tensor x, torch::Tensor weight) {
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = weight.size(1);
    auto out = torch::zeros({batch_size, out_features}, x.options());

    const int block_size_x = 32;
    const int block_size_y = 32;
    const int num_blocks_x = (out_features + block_size_x - 1) / block_size_x;
    const int num_blocks_y = (batch_size + block_size_y - 1) / block_size_y;

    gemm_max_sub_gelu_kernel<<<num_blocks_y, num_blocks_x, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), batch_size, in_features, out_features);

    return out;
}
"""

gemm_max_sub_gelu_cpp_source = (
    "torch::Tensor gemm_max_sub_gelu_cuda(torch::Tensor x, torch::Tensor weight);"
)

# Compile the inline CUDA code for GEMM and subsequent operations
gemm_max_sub_gelu = load_inline(
    name="gemm_max_sub_gelu",
    cpp_sources=gemm_max_sub_gelu_cpp_source,
    cuda_sources=gemm_max_sub_gelu_source,
    functions=["gemm_max_sub_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.max_dim = max_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        x = gemm_max_sub_gelu.gemm_max_sub_gelu_cuda(x, self.weight)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, max_dim]