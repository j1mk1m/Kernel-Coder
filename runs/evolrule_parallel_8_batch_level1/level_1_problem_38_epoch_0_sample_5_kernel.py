import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void l1_norm_sum(const float* x, float* sums, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    __shared__ float shared_partial[256];
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int j = tid; j < dim; j += blockDim.x) {
        float val = x[row * dim + j];
        sum += fabsf(val);
    }

    shared_partial[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_partial[tid] += shared_partial[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sums[row] = shared_partial[0];
    }
}

__global__ void l1_norm_normalize(const float* x, float* out, const float* sums, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    __shared__ float inv;
    if (threadIdx.x == 0) {
        inv = dim / sums[row];
    }
    __syncthreads();

    int tid = threadIdx.x;
    for (int j = tid; j < dim; j += blockDim.x) {
        int idx = row * dim + j;
        out[idx] = x[idx] * inv;
    }
}

extern "C" {
    torch::Tensor l1_norm_sum_cuda(torch::Tensor x, int batch_size, int dim) {
        auto sums = torch::empty({batch_size}, x.options());
        const int block_size = 256;
        const int num_blocks = batch_size;
        l1_norm_sum<<<num_blocks, block_size>>>(x.data_ptr<float>(), sums.data_ptr<float>(), batch_size, dim);
        return sums;
    }

    torch::Tensor l1_norm_normalize_cuda(torch::Tensor x, torch::Tensor sums, int batch_size, int dim) {
        auto out = torch::empty_like(x);
        const int block_size = 256;
        const int num_blocks = batch_size;
        l1_norm_normalize<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), sums.data_ptr<float>(), batch_size, dim);
        return out;
    }
}
"""

cpp_source = """
extern "C" {
    torch::Tensor l1_norm_sum_cuda(torch::Tensor x, int batch_size, int dim);
    torch::Tensor l1_norm_normalize_cuda(torch::Tensor x, torch::Tensor sums, int batch_size, int dim);
}
"""

# Compile the CUDA code
l1_norm_cuda = load_inline(
    name="l1_norm_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["l1_norm_sum_cuda", "l1_norm_normalize_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cuda_kernels = l1_norm_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move input to CUDA and process
        x_cuda = x.cuda()
        batch_size, dim = x_cuda.size()
        sums = self.cuda_kernels.l1_norm_sum_cuda(x_cuda, batch_size, dim)
        out_cuda = self.cuda_kernels.l1_norm_normalize_cuda(x_cuda, sums, batch_size, dim)
        # Return result to CPU to match original API
        return out_cuda.cpu()