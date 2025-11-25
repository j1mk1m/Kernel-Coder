import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

# C++ declarations for the custom CUDA function
layernorm_cpp_source = """
#include <torch/extension.h>

torch::Tensor custom_layernorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps);
"""

# CUDA kernel and implementation
layernorm_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    float* __restrict__ out,
    int batch_size,
    int features,
    int dim1,
    int dim2) {

    int sample_idx = blockIdx.x;
    int normalized_size = features * dim1 * dim2;
    int sample_offset = sample_idx * normalized_size;

    extern __shared__ float shared_mem[];
    float* s_sum = shared_mem;
    float* s_sum_sq = shared_mem + blockDim.x;

    int tid = threadIdx.x;

    // Initialize shared memory
    s_sum[tid] = 0.0f;
    s_sum_sq[tid] = 0.0f;
    __syncthreads();

    // Load data into shared memory for reduction
    for (int idx = tid; idx < normalized_size; idx += blockDim.x) {
        int global_idx = sample_offset + idx;
        float val = x[global_idx];
        s_sum[tid] += val;
        s_sum_sq[tid] += val * val;
    }
    __syncthreads();

    // Perform reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    // Compute mean and inv_std
    float mean, inv_std;
    if (tid == 0) {
        float sum = s_sum[0];
        float sum_sq = s_sum_sq[0];
        mean = sum / normalized_size;
        float var = (sum_sq / normalized_size) - mean * mean;
        inv_std = 1.0f / sqrtf(var + eps);

        // Store in shared memory
        s_sum[0] = mean;
        s_sum_sq[0] = inv_std;
    }
    __syncthreads();

    mean = s_sum[0];
    inv_std = s_sum_sq[0];

    // Normalize each element
    for (int idx = tid; idx < normalized_size; idx += blockDim.x) {
        int global_idx = sample_offset + idx;
        int f = idx / (dim1 * dim2);
        int rem = idx % (dim1 * dim2);
        int d1_idx = rem / dim2;
        int d2_idx = rem % dim2;
        int wb_idx = f * dim1 * dim2 + d1_idx * dim2 + d2_idx;

        float gamma = weight[wb_idx];
        float beta = bias[wb_idx];
        float val = x[global_idx];
        out[global_idx] = (val - mean) * inv_std * gamma + beta;
    }
}

torch::Tensor custom_layernorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    int batch_size = x.size(0);
    int features = x.size(1);
    int dim1 = x.size(2);
    int dim2 = x.size(3);
    int normalized_size = features * dim1 * dim2;

    auto out = torch::empty_like(x);

    int block_size = 256;
    int shared_size = 2 * block_size * sizeof(float);

    dim3 grid(batch_size);
    dim3 block(block_size);
    layernorm_kernel<<<grid, block, shared_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        eps,
        out.data_ptr<float>(),
        batch_size,
        features,
        dim1,
        dim2
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return out;
}
"""

# Load the custom CUDA extension
custom_layernorm = load_inline(
    name="custom_layernorm",
    cpp_sources=[layernorm_cpp_source],
    cuda_sources=[layernorm_cuda_source],
    functions=["custom_layernorm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5  # Default epsilon as in PyTorch's LayerNorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_layernorm.custom_layernorm_cuda(x, self.weight, self.bias, self.eps)