import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void layer_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int batch_size,
    int elements_per_sample,
    float epsilon
) {
    int sample_idx = blockIdx.x;
    int input_offset = sample_idx * elements_per_sample;
    int output_offset = input_offset;

    extern __shared__ float shared_sums[];

    int tid = threadIdx.x;

    // Initialize shared memory to zero
    if (tid < 2 * blockDim.x) {
        shared_sums[tid] = 0.0f;
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Accumulate partial sums
    for (int i = tid; i < elements_per_sample; i += blockDim.x) {
        float x = input[input_offset + i];
        local_sum += x;
        local_sum_sq += x * x;
    }

    // Write to shared memory
    shared_sums[tid] = local_sum;
    shared_sums[blockDim.x + tid] = local_sum_sq;
    __syncthreads();

    // Reduce sums
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
            shared_sums[blockDim.x + tid] += shared_sums[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    // Compute mean and variance
    float total_sum = shared_sums[0];
    float total_sum_sq = shared_sums[blockDim.x];
    float mean = total_sum / elements_per_sample;
    float variance = (total_sum_sq / elements_per_sample) - mean * mean;
    variance += epsilon;
    float inv_std = rsqrtf(variance);

    // Apply normalization and affine transform
    for (int i = tid; i < elements_per_sample; i += blockDim.x) {
        float x = input[input_offset + i];
        float x_centered = x - mean;
        float normalized = x_centered * inv_std;
        float gamma_val = gamma[i];
        float beta_val = beta[i];
        output[output_offset + i] = normalized * gamma_val + beta_val;
    }
}

torch::Tensor layer_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float epsilon
) {
    auto device = input.device();
    assert(device.type() == torch::kCUDA);

    int elements_per_sample = gamma.numel();
    int batch_size = input.size(0);
    assert(input.sizes().slice(1) == gamma.sizes());

    auto gamma_flat = gamma.view({-1});
    auto beta_flat = beta.view({-1});
    auto output = torch::empty_like(input);

    const int threads_per_block = 256;
    const int blocks_per_grid = batch_size;

    const int shared_mem_size = 2 * threads_per_block * sizeof(float);

    layer_norm_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        gamma_flat.data_ptr<float>(),
        beta_flat.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        elements_per_sample,
        epsilon
    );

    return output;
}
"""

layer_norm = load_inline(
    name="layer_norm_cuda",
    cuda_sources=layer_norm_source,
    functions=["layer_norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(*normalized_shape).cuda())
        self.bias = nn.Parameter(torch.zeros(*normalized_shape).cuda())
        self.epsilon = 1e-5

    def forward(self, x):
        return layer_norm.layer_norm_cuda(x, self.weight, self.bias, self.epsilon)