import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

triplet_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void triplet_loss_forward_kernel(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* loss_contrib,
    int batch_size,
    int dim,
    float margin) 
{
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    scalar_t d_ap_sq = 0.0;
    scalar_t d_an_sq = 0.0;

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        int idx = sample_idx * dim + d;
        scalar_t a_val = anchor[idx];
        scalar_t p_val = positive[idx];
        scalar_t n_val = negative[idx];

        scalar_t diff_ap = a_val - p_val;
        d_ap_sq += diff_ap * diff_ap;

        scalar_t diff_an = a_val - n_val;
        d_an_sq += diff_an * diff_an;
    }

    __shared__ scalar_t shared_ap[512];
    __shared__ scalar_t shared_an[512];

    shared_ap[threadIdx.x] = d_ap_sq;
    shared_an[threadIdx.x] = d_an_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>=1) {
        if (threadIdx.x < s) {
            shared_ap[threadIdx.x] += shared_ap[threadIdx.x + s];
            shared_an[threadIdx.x] += shared_an[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scalar_t final_d_ap = sqrt(shared_ap[0]);
        scalar_t final_d_an = sqrt(shared_an[0]);

        scalar_t term = final_d_ap - final_d_an + margin;
        if (term > 0) {
            loss_contrib[sample_idx] = term * term;
        } else {
            loss_contrib[sample_idx] = 0.0;
        }
    }
}

template <typename scalar_t>
__global__ void sum_reduction_kernel(
    const scalar_t* __restrict__ data,
    int n,
    scalar_t* result) 
{
    extern __shared__ scalar_t sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = 0;

    if (i < n)
        sdata[tid] = data[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(result, sdata[0]);
}

std::tuple<torch::Tensor> triplet_loss_forward(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) 
{
    const auto batch_size = anchor.size(0);
    const auto dim = anchor.size(1);

    auto loss_contrib = torch::empty({batch_size}, anchor.options());

    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;

    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_loss_forward", ([&] {
        triplet_loss_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            loss_contrib.data_ptr<scalar_t>(),
            batch_size,
            dim,
            margin
        );
    }));

    auto total_loss = torch::zeros(1, anchor.options());

    const int block_size = 1024;
    const int grid_size = (batch_size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "sum_reduction", ([&] {
        sum_reduction_kernel<scalar_t><<<grid_size, block_size, block_size * sizeof(scalar_t)>>>(
            loss_contrib.data_ptr<scalar_t>(),
            batch_size,
            total_loss.data_ptr<scalar_t>()
        );
    }));

    total_loss = total_loss.div(batch_size);

    return std::make_tuple(total_loss);
}
"""

triplet_loss_cpp_source = """
std::tuple<torch::Tensor> triplet_loss_forward(torch::Tensor, torch::Tensor, torch::Tensor, float);
"""

triplet_loss = load_inline(
    name="triplet_loss",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["triplet_loss_forward"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triplet_loss.triplet_loss_forward(anchor, positive, negative, self.margin)[0]