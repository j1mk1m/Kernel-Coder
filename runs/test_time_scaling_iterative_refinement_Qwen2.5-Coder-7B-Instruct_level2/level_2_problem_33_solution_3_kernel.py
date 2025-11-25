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
    dim3 grid((n + block_size - 1) / block_size, (m + block_size - 1) / block_size);
    dim3 block(block_size, block_size);

    gemm_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

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


# Define the custom CUDA kernel for batch normalization
bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bn_kernel(const float* x, const float* mean, const float* var, const float* gamma, const float* beta, float* y, int n, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float inv_std = 1.0f / sqrt(var[idx] + eps);
        y[idx] = gamma[idx] * (x[idx] - mean[idx]) * inv_std + beta[idx];
    }
}

torch::Tensor bn_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta) {
    auto n = x.numel();
    auto y = torch::zeros_like(x);

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    bn_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(), n, eps);

    return y;
}
"""

bn_cpp_source = (
    "torch::Tensor bn_cuda(torch::Tensor x, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta);"
)

# Compile the inline CUDA code for batch normalization
bn = load_inline(
    name="bn",
    cpp_sources=bn_cpp_source,
    cuda_sources=bn_source,
    functions=["bn_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = bn

    def forward(self, x):
        x = self.gemm.gemm_cuda(x, self.gemm.weight.t())
        x = x * self.scale
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, unbiased=False, keepdim=True)
        x = self.bn.bn_cuda(x, mean, var, self.scale, self.scale * 0.0)
        return x