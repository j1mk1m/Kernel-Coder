import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Your custom CUDA kernel code here
custom_gemm_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom Gemm and LeakyReLU kernel
__global__ void gemm_leakyrelu_kernel(float* a, float* b, float* c, float* out, int m, int n, int k, float alpha, float beta, float negative_slope) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum * alpha + beta;
        out[row * n + col] = fmaxf(c[row * n + col], negative_slope * c[row * n + col]);
    }
}

torch::Tensor gemm_leakyrelu_cuda(torch::Tensor a, torch::Tensor b, float alpha, float beta, float negative_slope) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto out = torch::zeros({m, n}, a.options());

    const int block_size = 32;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    gemm_leakyrelu_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), nullptr, out.data_ptr<float>(), m, n, k, alpha, beta, negative_slope);

    return out;
}
"""

custom_gemm_leakyrelu_cpp_source = (
    "torch::Tensor gemm_leakyrelu_cuda(torch::Tensor a, torch::Tensor b, float alpha, float beta, float negative_slope);"
)

# Compile the inline CUDA code for custom Gemm and LeakyReLU
custom_gemm_leakyrelu = load_inline(
    name="custom_gemm_leakyrelu",
    cpp_sources=custom_gemm_leakyrelu_cpp_source,
    cuda_sources=custom_gemm_leakyrelu_source,
    functions=["gemm_leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.custom_gemm_leakyrelu = custom_gemm_leakyrelu

    def forward(self, x):
        x = self.custom_gemm_leakyrelu.gemm_leakyrelu_cuda(x, x.t(), 1.0, 0.0, negative_slope)
        x = x * multiplier
        return x