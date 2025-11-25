import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for each operation

gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < m && col < n) {
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

void gemm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((n + threads_per_block.x - 1) / threads_per_block.x, (m + threads_per_block.y - 1) / threads_per_block.y);

    gemm_kernel<<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);
}
"""

division_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void division_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] /= 2.0f;
    }
}

void division_cuda(torch::Tensor x) {
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    division_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), size);
}
"""

summing_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void summing_kernel(const float* x, float* out, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = 0.0f;
    if (i < size) {
        sdata[tid] = x[i];
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

void summing_cuda(torch::Tensor x, torch::Tensor out) {
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    summing_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
}
"""

scaling_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scaling_kernel(float* x, float factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] *= factor;
    }
}

void scaling_cuda(torch::Tensor x, float factor) {
    auto size = x.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scaling_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), factor, size);
}
"""

# Compile the inline CUDA code for each operation
gemm = load_inline(
    name="gemm",
    cpp_sources="void gemm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c);",
    cuda_sources=gemm_source,
    functions=["gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

division = load_inline(
    name="division",
    cpp_sources="void division_cuda(torch::Tensor x);",
    cuda_sources=division_source,
    functions=["division_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

summing = load_inline(
    name="summing",
    cpp_sources="void summing_cuda(torch::Tensor x, torch::Tensor out);",
    cuda_sources=summing_source,
    functions=["summing_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

scaling = load_inline(
    name="scaling",
    cpp_sources="void scaling_cuda(torch::Tensor x, float factor);",
    cuda_sources=scaling_source,
    functions=["scaling_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        y = torch.zeros((x.size(0), self.weight.size(0)), device=x.device)
        gemm.gemm_cuda(x, self.weight.T, y)

        division.division_cuda(y)

        z = torch.zeros(1, device=y.device)
        summing.summing_cuda(y, z)

        scaled_output = torch.zeros_like(z)
        scaling.scaling_cuda(scaled_output, self.scaling_factor)

        return scaled_output


batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]