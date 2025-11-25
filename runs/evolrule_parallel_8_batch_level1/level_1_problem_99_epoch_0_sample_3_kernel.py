import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

triplet_loss_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__global__ void compute_loss_kernel(
    const T* __restrict__ anchor, const T* __restrict__ positive, const T* __restrict__ negative,
    T* __restrict__ loss_out, int batch_size, int dim, T margin) {
    int i = blockIdx.x;
    if (i >= batch_size) return;

    extern __shared__ T shared[];
    T* d_p_partial = shared;
    T* d_n_partial = shared + blockDim.x;

    int tid = threadIdx.x;
    T sum_p = 0.0;
    T sum_n = 0.0;

    for (int k = tid; k < dim; k += blockDim.x) {
        T a_val = anchor[i * dim + k];
        T p_val = positive[i * dim + k];
        T n_val = negative[i * dim + k];
        T diff_p = a_val - p_val;
        sum_p += diff_p * diff_p;
        T diff_n = a_val - n_val;
        sum_n += diff_n * diff_n;
    }

    d_p_partial[tid] = sum_p;
    d_n_partial[tid] = sum_n;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            d_p_partial[tid] += d_p_partial[tid + s];
            d_n_partial[tid] += d_n_partial[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        T d_p = d_p_partial[0];
        T d_n = d_n_partial[0];
        T loss_i = fmax(0.0, d_p - d_n + margin);
        loss_out[i] = loss_i;
    }
}

template <typename T>
__global__ void compute_partial_sums(
    const T* __restrict__ data, T* __restrict__ partial_sums, int size) {
    extern __shared__ T shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;

    T sum = 0.0;
    for (int i = bid * block_size + tid; i < size; i += gridDim.x * block_size) {
        sum += data[i];
    }

    shared[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[bid] = shared[0];
    }
}

torch::Tensor compute_triplet_loss_cuda(
    torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative) {

    int batch_size = anchor.size(0);
    int dim = anchor.size(1);

    auto a = anchor.contiguous();
    auto p = positive.contiguous();
    auto n = negative.contiguous();

    auto loss_out = torch::empty({batch_size}, a.options());

    int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    size_t shared_size = 2 * block_size * sizeof(float);
    compute_loss_kernel<float><<<grid, block, shared_size>>>(
        a.data_ptr<float>(), p.data_ptr<float>(), n.data_ptr<float>(),
        loss_out.data_ptr<float>(), batch_size, dim, 1.0f);

    int reduce_block_size = 1024;
    int reduce_num_blocks = (batch_size + reduce_block_size - 1) / reduce_block_size;
    auto partial_sums = torch::empty({reduce_num_blocks}, a.options());

    compute_partial_sums<float><<<reduce_num_blocks, reduce_block_size, reduce_block_size * sizeof(float)>>>(
        loss_out.data_ptr<float>(), partial_sums.data_ptr<float>(), batch_size);

    int N = reduce_num_blocks;
    int second_block_size = min(1024, N);
    int second_num_blocks = 1;

    auto total_loss = torch::zeros(1, a.options());

    compute_partial_sums<float><<<second_num_blocks, second_block_size, second_block_size * sizeof(float)>>>(
        partial_sums.data_ptr<float>(), total_loss.data_ptr<float>(), N);

    return total_loss / batch_size;
}
"""

triplet_loss_cpp_source = """
torch::Tensor compute_triplet_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative);
"""

triplet_loss = load_inline(
    name="triplet_loss",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["compute_triplet_loss_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, anchor, positive, negative):
        return triplet_loss.compute_triplet_loss_cuda(anchor, positive, negative)