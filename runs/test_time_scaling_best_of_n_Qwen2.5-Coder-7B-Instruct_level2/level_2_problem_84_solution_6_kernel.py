import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Gemm
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom Gemm kernel
__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);
    auto C = torch::zeros({M, N}, A.options());

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);

    gemm_kernel<<<blocks_per_grid, threads_per_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

# Custom CUDA kernel for Batch Normalization
bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom Batch Normalization kernel
__global__ void batch_norm_kernel(const float* x, const float* mean, const float* var, float* y, float* gamma, float* beta, float eps, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * D) {
        int d = idx % D;
        y[idx] = gamma[d] * (x[idx] - mean[d]) / sqrt(var[d] + eps) + beta[d];
    }
}

torch::Tensor batch_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto N = x.size(0);
    auto D = x.size(1);
    auto mean = torch::mean(x, {0});
    auto var = torch::var_mean(x, {0}).second;
    auto y = torch::zeros_like(x);

    dim3 threads_per_block(256);
    dim3 blocks_per_grid((N * D + threads_per_block.x - 1) / threads_per_block.x);

    batch_norm_kernel<<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), y.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), eps, N, D);

    return y;
}
"""

# Custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom Softmax kernel
__global__ void softmax_kernel(const float* x, float* y, int N, int D) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N * D) {
        sdata[tid] = exp(x[i]);
        __syncthreads();

        int offset = blockDim.x;
        while (offset > 0) {
            if (tid < offset) {
                sdata[tid] += sdata[tid + offset];
            }
            __syncthreads();
            offset /= 2;
        }

        if (tid == 0) {
            y[i] = sdata[0];
        }
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto N = x.size(0);
    auto D = x.size(1);
    auto y = torch::zeros_like(x);

    dim3 threads_per_block(256);
    dim3 blocks_per_grid((N * D + threads_per_block.x - 1) / threads_per_block.x);

    softmax_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(x.data_ptr<float>(), y.data_ptr<float>(), N, D);

    return y;
}
"""

# Load the custom CUDA kernels
gemm = load_inline(name="gemm", cpp_sources="", cuda_sources=gemm_source, functions=["gemm_cuda"], verbose=True)
bn = load_inline(name="bn", cpp_sources="", cuda_sources=bn_source, functions=["batch_norm_cuda"], verbose=True)
softmax = load_inline(name="softmax", cpp_sources="", cuda_sources=softmax_source, functions=["softmax_cuda"], verbose=True)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.bn = bn
        self.scale = nn.Parameter(torch.ones(scale_shape))

    def forward(self, x):
        x = self.gemm.gemm_cuda(x, self.gemm.weight)
        x = self.bn.batch_norm_cuda(x, self.scale, self.scale, bn_eps)
        x = self.softmax.softmax_cuda(x)
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    bn_eps = 1e-5
    bn_momentum = 0.1
    scale_shape = (1,)

    model = ModelNew(in_features, out_features, bn_eps, bn_momentum, scale_shape)
    inputs = get_inputs()[0].cuda()
    outputs = model(inputs)
    print(outputs.shape)