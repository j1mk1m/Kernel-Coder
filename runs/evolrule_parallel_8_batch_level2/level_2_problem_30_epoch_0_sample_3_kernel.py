import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the fused CUDA kernel for GEMM + GroupNorm + HardTanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void fused_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups,
    float eps,
    float min_val,
    float max_val) {

    const int B = blockIdx.x;
    const int tid = threadIdx.x;

    // Each thread processes a channel
    const int out_channel = tid;
    if (out_channel >= out_features) return;

    // Compute GEMM: y = input @ weight^T + bias
    scalar_t pre_act = bias[out_channel];
    for (int i = 0; i < in_features; ++i) {
        pre_act += input[B * in_features + i] * weight[out_channel * in_features + i];
    }

    // Compute group parameters
    const int group_size = out_features / num_groups;
    const int group_id = out_channel / group_size;
    const int group_start = group_id * group_size;
    const int group_end = group_start + group_size;

    // Shared memory for group mean and var
    __shared__ scalar_t shared_sum[32]; // Enough for groups up to 32
    __shared__ scalar_t shared_sqsum[32];

    if (tid < group_size) {
        shared_sum[group_id] = 0;
        shared_sqsum[group_id] = 0;
    }
    __syncthreads();

    // Compute sum and squared sum for the group
    scalar_t val = (out_channel >= group_start && out_channel < group_end) ? pre_act : 0;
    atomicAdd(&shared_sum[group_id], val);
    atomicAdd(&shared_sqsum[group_id], val * val);

    __syncthreads();

    // Only one thread per group computes mean and variance
    if (tid == 0 && group_id == group_id) {
        scalar_t mean = shared_sum[group_id] / group_size;
        scalar_t sq_mean = shared_sqsum[group_id] / group_size;
        scalar_t var = sq_mean - mean * mean;
        var = max(var, scalar_t(eps));

        // Broadcast mean and var to all threads in the group
        // This part requires a way to propagate to all threads, which is non-trivial
        // For simplicity, we'll recompute here for each channel in the group (inefficient but illustrative)
    }

    // Re-fetch mean and var (inefficient, but for example purposes)
    scalar_t mean = shared_sum[group_id] / group_size;
    scalar_t var = (shared_sqsum[group_id] / group_size) - mean * mean;
    var = fmax(var, eps);

    // Normalize
    scalar_t normalized = (pre_act - mean) / sqrt(var);
    normalized = normalized * gamma[out_channel] + beta[out_channel];

    // Apply HardTanh
    scalar_t clamped = normalized < min_val ? min_val : (normalized > max_val ? max_val : normalized);

    // Write output
    if (out_channel < out_features) {
        output[B * out_features + out_channel] = clamped;
    }
}

// This kernel is still not fully optimized, but shows the approach

torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps,
    float min_val,
    float max_val) {

    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int threads = 256;
    const dim3 blocks(batch_size, 1, 1);

    // Launch kernel
    fused_forward_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_features, out_features,
        num_groups, eps, min_val, max_val);

    return output;
}
"""

# Compile the fused CUDA kernel
fused_mod = load_inline(
    name="fused_mod",
    cuda_sources=fused_kernel_source,
    functions=["fused_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gamma = nn.Parameter(torch.empty(out_features))
        self.beta = nn.Parameter(torch.empty(out_features))
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.eps = 1e-5  # Default epsilon for GroupNorm

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        return fused_mod.fused_forward(
            x, self.weight, self.bias, self.gamma, self.beta,
            self.num_groups, self.eps, self.hardtanh_min, self.hardtanh_max
        )