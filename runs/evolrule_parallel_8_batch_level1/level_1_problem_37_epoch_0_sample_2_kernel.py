import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source code for the custom operator
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_squares(const float* x, float* partial_sums, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_threads = blockDim.x;

    float sum = 0.0f;
    for (int i = bid * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x) {
        float val = x[i];
        sum += val * val;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[bid] = sdata[0];
    }
}

__global__ void final_reduce(float* partial_sums, int num_partial, float* total_sum) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    if (tid < num_partial) {
        sdata[tid] = partial_sums[tid];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Reduce within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *total_sum = sdata[0];
    }
}

__global__ void divide_kernel(const float* x, float* out, float norm, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = x[idx] / norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor x) {
    int N = x.numel();
    const int grid_size = 1024;
    const int block_size = 256;
    const int block_size_final = 1024;

    // Allocate partial_sums on device
    auto partial_sums = torch::empty({grid_size}, torch::CUDA(x.device()));

    // Launch first kernel
    reduce_squares<<<grid_size, block_size, block_size * sizeof(float)>>>(
        x.data_ptr<float>(), partial_sums.data_ptr<float>(), N
    );

    // Allocate total_sum
    auto total_sum = torch::zeros({1}, torch::CUDA(x.device()));

    // Launch second kernel with block_size_final and shared memory for num_partial elements
    final_reduce<<<1, block_size_final, grid_size * sizeof(float)>>>(
        partial_sums.data_ptr<float>(), grid_size, total_sum.data_ptr<float>()
    );

    // Compute norm. Since total_sum is a tensor on device, we need to get its value.
    float norm_val = sqrt(total_sum.item<float>());

    // Create output tensor
    auto out = torch::empty_like(x);

    // Launch division kernel
    const int threads_div = 256;
    const int blocks_div = (N + threads_div - 1) / threads_div;
    divide_kernel<<<blocks_div, threads_div>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), norm_val, N
    );

    return out;
}
"""

cpp_source = "torch::Tensor frobenius_norm_cuda(torch::Tensor x);"

# Compile the CUDA code
frobenius_norm_cuda = load_inline(
    name="frobenius_norm",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["frobenius_norm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.frobenius_norm_cuda = frobenius_norm_cuda

    def forward(self, x):
        return self.frobenius_norm_cuda.frobenius_norm_cuda(x)

def get_inputs():
    batch_size = 112
    features = 64
    dim1 = 512
    dim2 = 512
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return []