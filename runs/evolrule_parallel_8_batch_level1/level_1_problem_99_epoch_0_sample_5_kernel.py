import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

triplet_loss_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void triplet_loss_kernel(
    const scalar_t* __restrict__ a_data,
    const scalar_t* __restrict__ p_data,
    const scalar_t* __restrict__ n_data,
    scalar_t margin,
    scalar_t* loss_per_sample,
    int batch_size,
    int dim
) {
    int i = blockIdx.x;
    if (i >= batch_size) return;

    scalar_t norm_sq_ap = 0.0;
    scalar_t norm_sq_an = 0.0;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int d = tid; d < dim; d += num_threads) {
        scalar_t a_val = a_data[i * dim + d];
        scalar_t p_val = p_data[i * dim + d];
        scalar_t n_val = n_data[i * dim + d];

        scalar_t diff_ap = a_val - p_val;
        norm_sq_ap += diff_ap * diff_ap;

        scalar_t diff_an = a_val - n_val;
        norm_sq_an += diff_an * diff_an;
    }

    extern __shared__ scalar_t shared[];
    scalar_t* s_norm_sq_ap = shared;
    scalar_t* s_norm_sq_an = shared + blockDim.x;

    s_norm_sq_ap[tid] = norm_sq_ap;
    s_norm_sq_an[tid] = norm_sq_an;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_norm_sq_ap[tid] += s_norm_sq_ap[tid + s];
            s_norm_sq_an[tid] += s_norm_sq_an[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        scalar_t d_ap = sqrt(s_norm_sq_ap[0]);
        scalar_t d_an = sqrt(s_norm_sq_an[0]);

        scalar_t term = d_ap - d_an + margin;
        scalar_t loss_i = (term > 0.0) ? term : 0.0;

        loss_per_sample[i] = loss_i;
    }
}

template <typename scalar_t>
__global__ void sum_reduction_kernel(
    scalar_t* loss_per_sample,
    scalar_t* total_loss,
    int n,
    int block_size
) {
    extern __shared__ scalar_t shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * block_size + tid;
    scalar_t sum = 0.0;

    if (i < n) {
        sum = loss_per_sample[i];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total_loss, shared[0]);
    }
}

at::Tensor triplet_loss_cuda(
    at::Tensor a,
    at::Tensor p,
    at::Tensor n,
    float margin
) {
    const int batch_size = a.size(0);
    const int dim = a.size(1);

    auto loss_per_sample = at::empty({batch_size}, a.options()).cuda();
    auto total_loss = at::zeros({1}, a.options()).cuda();

    int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    size_t shared_size = 2 * block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "triplet_loss_cuda", ([&] {
        triplet_loss_kernel<scalar_t><<<grid, block, shared_size, at::cuda::getCurrentCUDAStream()>>>(
            a.data_ptr<scalar_t>(),
            p.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            margin,
            loss_per_sample.data_ptr<scalar_t>(),
            batch_size,
            dim
        );
    }));

    int reduction_block_size = 256;
    int reduction_grid_size = (batch_size + reduction_block_size - 1) / reduction_block_size;
    dim3 reduction_grid(reduction_grid_size);
    dim3 reduction_block(reduction_block_size);
    size_t reduction_shared_size = reduction_block_size * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "sum_reduction_cuda", ([&] {
        sum_reduction_kernel<scalar_t><<<reduction_grid, reduction_block, reduction_shared_size, at::cuda::getCurrentCUDAStream()>>>(
            loss_per_sample.data_ptr<scalar_t>(),
            total_loss.data_ptr<scalar_t>(),
            batch_size,
            reduction_block_size
        );
    }));

    auto avg_loss = total_loss / static_cast<float>(batch_size);

    return avg_loss;
}
"""

triplet_loss_header = """
at::Tensor triplet_loss_cuda(
    at::Tensor a,
    at::Tensor p,
    at::Tensor n,
    float margin
);
"""

triplet_loss_cuda = load_inline(
    name="triplet_loss_cuda",
    cpp_sources=triplet_loss_header,
    cuda_sources=triplet_loss_source,
    functions=["triplet_loss_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triplet_loss_cuda.triplet_loss_cuda(anchor, positive, negative, self.margin)

batch_size = 32768
input_shape = (8192,)

def get_inputs():
    scale = torch.rand(())
    return [
        torch.rand(batch_size, *input_shape)*scale,
        torch.rand(batch_size, *input_shape),
        torch.rand(batch_size, *input_shape)
    ]

def get_init_inputs():
    return [1.0]  # Default margin