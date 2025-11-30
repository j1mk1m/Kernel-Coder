import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for GEMM, scaling, hardtanh, and GELU
gemm_scaling_hardtanh_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_scaling_hardtanh_gelu_kernel(const float* A, const float* B, float* C, int M, int N, int K, float scale, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int i = idx / N;
        int j = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[idx] = fmaxf(fminf(sum * scale, max_val), min_val) * 0.5f * (1.0f + tanh(sqrt(2.0f / M_PI) * (sum * scale + 0.044715f * pow(sum * scale, 3.0f))));
    }
}

torch::Tensor gemm_scaling_hardtanh_gelu_cuda(torch::Tensor A, torch::Tensor B, float scale, float min_val, float max_val) {
    auto M = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);
    auto C = torch::zeros({M, N}, A.options());

    const int block_size = 256;
    const int num_blocks = (M * N + block_size - 1) / block_size;

    gemm_scaling_hardtanh_gelu_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, scale, min_val, max_val);

    return C;
}
"""

gemm_scaling_hardtanh_gelu_cpp_source = (
    "torch::Tensor gemm_scaling_hardtanh_gelu_cuda(torch::Tensor A, torch::Tensor B, float scale, float min_val, float max_val);"
)

# Compile the inline CUDA code for GEMM, scaling, hardtanh, and GELU
gemm_scaling_hardtanh_gelu = load_inline(
    name="gemm_scaling_hardtanh_gelu",
    cpp_sources=gemm_scaling_hardtanh_gelu_cpp_source,
    cuda_sources=gemm_scaling_hardtanh_gelu_source,
    functions=["gemm_scaling_hardtanh_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm_scaling_hardtanh_gelu = gemm_scaling_hardtanh_gelu

    def forward(self, x):
        x = self.gemm_scaling_hardtanh_gelu.gemm_scaling_hardtanh_gelu_cuda(x, x.t(), scaling_factor, hardtanh_min, hardtanh_max)
        return x