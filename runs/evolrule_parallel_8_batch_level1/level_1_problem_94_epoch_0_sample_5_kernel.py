import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA code for custom MSE loss
mse_loss_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void compute_partial_sums(
    const float* a, const float* b, float* partial_sums, int N) {

    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int start = bid * blockDim.x;
    int end = min(start + blockDim.x, N);

    float sum = 0.0f;

    for (int i = start + tid; i < end; i += blockDim.x) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>=1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[bid] = shared[0];
    }
}

__global__ void reduce_partial_sums(
    const float* partial_sums, float* total_sum, int num_partial) {

    extern __shared__ float shared[];

    int tid = threadIdx.x;

    float sum = 0.0f;

    for (int i = tid; i < num_partial; i += blockDim.x) {
        sum += partial_sums[i];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>=1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total_sum, shared[0]);
    }
}

at::Tensor compute_mse_loss_cuda(const at::Tensor& a, const at::Tensor& b) {
    const int total_elements = a.numel();
    const int block_size1 = 256;
    const dim3 grid_size1((total_elements + block_size1 - 1) / block_size1);
    const dim3 block_size1_d(block_size1);

    auto partial_sums = at::empty({grid_size1.x}, a.options());

    compute_partial_sums<<<grid_size1, block_size1_d, block_size1 * sizeof(float)>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), partial_sums.data_ptr<float>(), total_elements);

    const int block_size2 = 4096;
    auto total_sum = at::zeros({1}, a.options());

    reduce_partial_sums<<<1, block_size2, block_size2 * sizeof(float)>>>(
        partial_sums.data_ptr<float>(), total_sum.data_ptr<float>(), grid_size1.x);

    total_sum /= total_elements;

    return total_sum;
}
"""

# Load the CUDA extension
mse_loss_cuda = load_inline(
    name="mse_loss_cuda",
    cpp_sources="",
    cuda_sources=mse_loss_cuda_source,
    functions=["compute_mse_loss_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        return mse_loss_cuda.compute_mse_loss_cuda(predictions, targets).squeeze()