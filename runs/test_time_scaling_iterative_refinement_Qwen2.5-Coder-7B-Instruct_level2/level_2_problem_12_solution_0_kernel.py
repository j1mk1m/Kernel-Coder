import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Gemm, multiplication, and LeakyReLU
gemm_mul_leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Implement the GEMM operation using CUDA
__global__ void gemm_mul_leakyrelu_kernel(const float* A, const float* B, float* C, int M, int N, int K, float alpha) {
    // Your implementation here
}

torch::Tensor gemm_mul_leakyrelu_cuda(torch::Tensor A, torch::Tensor B, float alpha) {
    // Your implementation here
}
"""

gemm_mul_leakyrelu_cpp_source = (
    "torch::Tensor gemm_mul_leakyrelu_cuda(torch::Tensor A, torch::Tensor B, float alpha);"
)

# Compile the inline CUDA code for Gemm, multiplication, and LeakyReLU
gemm_mul_leakyrelu = load_inline(
    name="gemm_mul_leakyrelu",
    cpp_sources=gemm_mul_leakyrelu_cpp_source,
    cuda_sources=gemm_mul_leakyrelu_source,
    functions=["gemm_mul_leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm_mul_leakyrelu = gemm_mul_leakyrelu
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x):
        x = self.gemm_mul_leakyrelu.gemm_mul_leakyrelu_cuda(x, x.t(), self.negative_slope)  # Assuming x.t() is the transposed version of x
        x = x * self.multiplier
        return x