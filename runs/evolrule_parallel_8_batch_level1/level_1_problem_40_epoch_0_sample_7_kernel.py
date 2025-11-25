import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layer_norm_kernel(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    int normalized_size,
    int batch_stride,
    float epsilon
) {
    int sample_idx = blockIdx.x;
    int sample_offset = sample_idx * batch_stride;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float* shared = sdata;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < normalized_size; i += blockDim.x) {
        sum += x[sample_offset + i];
    }
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float total_sum = shared[0];
    float mean = total_sum / normalized_size;

    // Compute variance
    float sum_sq = 0.0f;
    for (int i = tid; i < normalized_size; i += blockDim.x) {
        float val = x[sample_offset + i] - mean;
        sum_sq += val * val;
    }
    shared[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float variance = shared[0] / normalized_size;
    float inv_std = rsqrt(variance + epsilon);

    // Compute output
    for (int i = tid; i < normalized_size; i += blockDim.x) {
        float val = x[sample_offset + i];
        float normed = (val - mean) * inv_std;
        normed *= gamma[i];
        normed += beta[i];
        y[sample_offset + i] = normed;
    }
}

torch::Tensor layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int normalized_size,
    int batch_stride,
    float epsilon
) {
    auto out = torch::empty_like(x);
    const int threads_per_block = 256;
    const int blocks_per_grid = x.size(0);
    const size_t shared_mem_size = threads_per_block * sizeof(float);

    layer_norm_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        normalized_size,
        batch_stride,
        epsilon
    );

    return out;
}
"""

layer_norm_cpp_source = """
#include <torch/extension.h>

torch::Tensor layer_norm_cuda(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    int normalized_size,
    int batch_stride,
    float epsilon
);
"""

# Compile the inline CUDA code for layer normalization
layer_norm = load_inline(
    name="layer_norm",
    cpp_sources=[layer_norm_cpp_source],
    cuda_sources=[layer_norm_source],
    functions=["layer_norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_size = 1
        for s in normalized_shape:
            self.normalized_size *= s
        self.batch_stride = self.normalized_size  # stride between samples

    def forward(self, x):
        epsilon = 1e-5  # same as PyTorch's default
        return layer_norm.layer_norm_cuda(
            x,
            self.weight,
            self.bias,
            self.normalized_size,
            self.batch_stride,
            epsilon
        )