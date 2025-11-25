import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 112
features = 64
dim1 = 512
dim2 = 512

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_sum_of_squares(const float* x, float* partial_sums, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * blockDim.x + tid;
    float sum = 0.0f;
    while (offset < size) {
        sum += x[offset] * x[offset];
        offset += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

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

__global__ void reduce_partial_sums(float* partial_sums, float* total_sum, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = bid * blockDim.x + tid;
    if (offset < n) {
        sdata[tid] = partial_sums[offset];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(total_sum, sdata[0]);
    }
}

__global__ void normalize_tensor(const float* x, float* out, float norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] / norm;
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor x) {
    const int size = x.numel();
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    // Compute partial sums per block
    auto partial_sums = torch::empty(grid_size, torch::dtype(torch::kFloat32).device(x.device()));
    compute_sum_of_squares<<<grid_size, block_size, block_size * sizeof(float)>>>(
        x.data_ptr<float>(), partial_sums.data_ptr<float>(), size);
    cudaDeviceSynchronize();

    // Reduce partial sums to total sum
    auto total_sum = torch::zeros(1, torch::dtype(torch::kFloat32).device(x.device()));
    const int reduce_block = 256;
    const int reduce_grid = (grid_size + reduce_block - 1) / reduce_block;
    reduce_partial_sums<<<reduce_grid, reduce_block, reduce_block * sizeof(float)>>>(
        partial_sums.data_ptr<float>(), total_sum.data_ptr<float>(), grid_size);
    cudaDeviceSynchronize();

    // Compute norm
    float norm_val = sqrt(total_sum.item<float>());

    // Normalize
    auto out = torch::empty_like(x);
    const int norm_block = 256;
    const int norm_grid = (size + norm_block - 1) / norm_block;
    normalize_tensor<<<norm_grid, norm_block>>>(x.data_ptr<float>(), out.data_ptr<float>(), norm_val, size);
    cudaDeviceSynchronize();

    return out;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor frobenius_norm_cuda(torch::Tensor x);
"""

# Compile the CUDA code
frobenius_norm_cuda = load_inline(
    name="frobenius_norm_cuda",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["frobenius_norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return frobenius_norm_cuda.frobenius_norm_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return []