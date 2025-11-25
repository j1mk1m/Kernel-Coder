import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);

    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 16;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    gemm_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for GEMM
gemm = load_inline(
    name="gemm",
    cpp_sources=gemm_cpp_source,
    cuda_sources=gemm_source,
    functions=["gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for scaling
scaling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scaling_kernel(float* x, const float* scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] *= scale[idx];
    }
}

void scaling_cuda(torch::Tensor x, const torch::Tensor scale) {
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scaling_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), scale.data_ptr<float>(), size);
}
"""

scaling_cpp_source = (
    "void scaling_cuda(torch::Tensor x, const torch::Tensor scale);"
)

# Compile the inline CUDA code for scaling
scaling = load_inline(
    name="scaling",
    cpp_sources=scaling_cpp_source,
    cuda_sources=scaling_source,
    functions=["scaling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for batch normalization
bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bn_forward_kernel(const float* x, const float* mean, const float* var, const float* gamma, const float* beta, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        y[idx] = gamma[idx] * (x[idx] - mean[idx]) / sqrt(var[idx] + 1e-5) + beta[idx];
    }
}

void bn_forward_cuda(const torch::Tensor x, const torch::Tensor mean, const torch::Tensor var, const torch::Tensor gamma, const torch::Tensor beta, torch::Tensor y) {
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    bn_forward_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(), size);
}
"""

bn_cpp_source = (
    "void bn_forward_cuda(const torch::Tensor x, const torch::Tensor mean, const torch::Tensor var, const torch::Tensor gamma, const torch::Tensor beta, torch::Tensor y);"
)

# Compile the inline CUDA code for batch normalization
bn = load_inline(
    name="bn",
    cpp_sources=bn_cpp_source,
    cuda_sources=bn_source,
    functions=["bn_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn_mean = torch.zeros(out_features)
        self.bn_var = torch.ones(out_features)
        self.bn_gamma = torch.ones(out_features)
        self.bn_beta = torch.zeros(out_features)
        self.bn = bn

    def forward(self, x):
        x = self.gemm.gemm_cuda(x, self.gemm.weight.t())
        scaling_cuda(x, self.scale)
        self.bn_forward_cuda(x, self.bn_mean, self.bn_var, self.bn_gamma, self.bn_beta, x)
        return x

    def bn_forward_cuda(self, x, mean, var, gamma, beta, y):
        self.bn.bn_forward_cuda(x, mean, var, gamma, beta, y)