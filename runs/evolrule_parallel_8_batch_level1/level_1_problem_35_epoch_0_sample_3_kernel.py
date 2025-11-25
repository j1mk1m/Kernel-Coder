import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5  # Use the same epsilon as PyTorch's default

        # Define the CUDA kernels
        group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void compute_group_stats_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ sums,
    scalar_t* __restrict__ sum_squares,
    int batch_size,
    int num_features,
    int num_groups,
    int dim1,
    int dim2,
    int features_per_group) {

    int group_idx = blockIdx.x;
    if (group_idx >= num_groups) return;

    int start_channel = group_idx * features_per_group;

    extern __shared__ scalar_t shared_mem[];
    scalar_t* s_sum = shared_mem;
    scalar_t* s_sum_sq = shared_mem + blockDim.x;

    int tid = threadIdx.x;

    s_sum[tid] = 0;
    s_sum_sq[tid] = 0;
    __syncthreads();

    int total_elements = batch_size * features_per_group * dim1 * dim2;
    int elements_per_thread = (total_elements + blockDim.x - 1) / blockDim.x;

    for (int i = tid * elements_per_thread; i < total_elements; i += blockDim.x) {
        int batch = i / (features_per_group * dim1 * dim2);
        int remainder = i % (features_per_group * dim1 * dim2);
        int channel_in_group = remainder / (dim1 * dim2);
        int pos = remainder % (dim1 * dim2);
        int d1 = pos / dim2;
        int d2 = pos % dim2;

        int channel = start_channel + channel_in_group;

        int offset = batch * num_features * dim1 * dim2 +
                     channel * dim1 * dim2 +
                     d1 * dim2 + d2;

        scalar_t x = input[offset];
        s_sum[tid] += x;
        s_sum_sq[tid] += x * x;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sums[group_idx] = s_sum[0];
        sum_squares[group_idx] = s_sum_sq[0];
    }
}

template <typename scalar_t>
__global__ void compute_mean_var(
    const scalar_t* sums,
    const scalar_t* sum_squares,
    scalar_t* mean,
    scalar_t* var,
    int num_groups,
    int features_per_group,
    int batch_size,
    int dim1,
    int dim2,
    float eps) {

    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < num_groups; g += blockDim.x * gridDim.x) {
        int total_elements = batch_size * features_per_group * dim1 * dim2;
        scalar_t s = sums[g];
        scalar_t s_sq = sum_squares[g];
        scalar_t m = s / total_elements;
        scalar_t v = s_sq / total_elements - m * m;
        v = fmaxf(v, eps);
        v = sqrtf(v);
        mean[g] = m;
        var[g] = v;
    }
}

template <typename scalar_t>
__global__ void group_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    int batch_size,
    int num_features,
    int dim1,
    int dim2) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_features * dim1 * dim2) return;

    int batch = idx / (num_features * dim1 * dim2);
    int remainder = idx % (num_features * dim1 * dim2);
    int channel = remainder / (dim1 * dim2);
    int pos = remainder % (dim1 * dim2);
    int d1 = pos / dim2;
    int d2 = pos % dim2;

    int features_per_group = num_features / num_groups;
    int group_idx = channel / features_per_group;

    scalar_t mu = mean[group_idx];
    scalar_t inv_std = 1.0 / var[group_idx];

    int offset = batch * num_features * dim1 * dim2 +
                 channel * dim1 * dim2 +
                 d1 * dim2 + d2;

    scalar_t x = input[offset];
    scalar_t normalized = (x - mu) * inv_std;

    int gamma_idx = channel;
    scalar_t scaled = normalized * gamma[gamma_idx] + beta[gamma_idx];

    output[offset] = scaled;
}

template <typename scalar_t>
at::Tensor group_norm_cuda(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    int num_groups,
    float eps) {

    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    const int dim1 = input.size(2);
    const int dim2 = input.size(3);
    const int features_per_group = num_features / num_groups;

    auto device = input.device();
    auto sums = at::empty({num_groups}, input.options());
    auto sum_squares = at::empty({num_groups}, input.options());
    auto mean = at::empty({num_groups}, input.options());
    auto var = at::empty({num_groups}, input.options());

    const int threads_per_block = 256;
    const int blocks_per_grid = num_groups;
    const int shared_size = 2 * threads_per_block * sizeof(scalar_t);

    compute_group_stats_kernel<scalar_t><<<blocks_per_grid, threads_per_block, shared_size>>>(
        input.data_ptr<scalar_t>(),
        sums.data_ptr<scalar_t>(),
        sum_squares.data_ptr<scalar_t>(),
        batch_size, num_features, num_groups, dim1, dim2, features_per_group
    );

    dim3 mv_blocks(1);
    dim3 mv_threads(256);
    compute_mean_var<scalar_t><<<mv_blocks, mv_threads>>>(
        sums.data_ptr<scalar_t>(),
        sum_squares.data_ptr<scalar_t>(),
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>(),
        num_groups,
        features_per_group,
        batch_size,
        dim1,
        dim2,
        eps
    );

    auto output = at::empty_like(input);

    const int total_elements = batch_size * num_features * dim1 * dim2;
    const int norm_threads = 1024;
    const int norm_blocks = (total_elements + norm_threads - 1) / norm_threads;

    group_norm_kernel<scalar_t><<<norm_blocks, norm_threads>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        gamma.data_ptr<scalar_t>(),
        beta.data_ptr<scalar_t>(),
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>(),
        batch_size,
        num_features,
        dim1,
        dim2
    );

    return output;
}

at::Tensor group_norm(
    at::Tensor input,
    at::Tensor gamma,
    at::Tensor beta,
    int num_groups,
    float eps) {
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(gamma.is_cuda(), "gamma must be a CUDA tensor");
    AT_ASSERTM(beta.is_cuda(), "beta must be a CUDA tensor");
    AT_ASSERTM(input.dim() == 4, "Input must be 4D tensor (batch, features, h, w)");

    auto dtype = input.scalar_type();
    if (dtype == at::ScalarType::Float) {
        return group_norm_cuda<float>(input, gamma, beta, num_groups, eps);
    } else {
        AT_ERROR("Unsupported tensor type");
    }
}
"""

        group_norm_cpp_source = """
extern "C" {
    torch::Tensor group_norm(
        torch::Tensor input,
        torch::Tensor gamma,
        torch::Tensor beta,
        int num_groups,
        float eps);
}
"""

        # Compile the inline CUDA code
        self.group_norm_cuda = load_inline(
            name="group_norm_cuda",
            cpp_sources=group_norm_cpp_source,
            cuda_sources=group_norm_source,
            functions=["group_norm"],
            verbose=True,
            extra_cflags=[""],
            extra_cuda_flags=["-lineinfo"],
            extra_ldflags=[""],
        )

    def forward(self, x):
        return self.group_norm_cuda.group_norm(
            x, self.gamma, self.beta, self.num_groups, self.eps
        )