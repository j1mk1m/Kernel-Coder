import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void log_softmax_kernel(const float* data, float* log_data, int batch_size, int feature_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * feature_dim) {
        return;
    }

    int row = idx / feature_dim;
    int col = idx % feature_dim;
    float max_val = -FLT_MAX;
    for (int j = 0; j < feature_dim; ++j) {
        max_val = fmaxf(max_val, data[row * feature_dim + j]);
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < feature_dim; ++j) {
        sum_exp += expf(data[row * feature_dim + j] - max_val);
    }

    log_data[idx] = data[row * feature_dim + col] - max_val - logf(sum_exp);
}

torch::Tensor log_softmax_cuda(torch::Tensor data) {
    auto batch_size = data.size(0);
    auto feature_dim = data.size(1);
    auto log_data = torch::zeros_like(data);

    const int block_size = 256;
    const int num_blocks = (batch_size * feature_dim + block_size - 1) / block_size;

    log_softmax_kernel<<<num_blocks, block_size>>>(data.data_ptr<float>(), log_data.data_ptr<float>(), batch_size, feature_dim);

    return log_data;
}

__global__ void exp_kernel(const float* data, float* exp_data, int batch_size, int feature_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * feature_dim) {
        return;
    }

    int row = idx / feature_dim;
    int col = idx % feature_dim;
    exp_data[idx] = expf(data[row * feature_dim + col]);
}

torch::Tensor exp_cuda(torch::Tensor data) {
    auto batch_size = data.size(0);
    auto feature_dim = data.size(1);
    auto exp_data = torch::zeros_like(data);

    const int block_size = 256;
    const int num_blocks = (batch_size * feature_dim + block_size - 1) / block_size;

    exp_kernel<<<num_blocks, block_size>>>(data.data_ptr<float>(), exp_data.data_ptr<float>(), batch_size, feature_dim);

    return exp_data;
}

__global__ void sum_kernel(const float* data, float* sum_data, int batch_size, int feature_dim) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size * feature_dim) {
        return;
    }

    sdata[tid] = data[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_data, sdata[0]);
    }
}

torch::Tensor sum_cuda(torch::Tensor data) {
    auto batch_size = data.size(0);
    auto feature_dim = data.size(1);
    auto sum_data = torch::zeros({batch_size}, device=data.device());

    const int block_size = 256;
    const int num_blocks = (batch_size * feature_dim + block_size - 1) / block_size;

    sum_kernel<<<num_blocks, block_size, sizeof(float) * block_size>>>(data.data_ptr<float>(), sum_data.data_ptr<float>(), batch_size, feature_dim);

    return sum_data;
}
"""

softmax_cpp_source = (
    "torch::Tensor log_softmax_cuda(torch::Tensor data);"
    "torch::Tensor exp_cuda(torch::Tensor data);"
    "torch::Tensor sum_cuda(torch::Tensor data);"
)

# Compile the inline CUDA code for Softmax
try:
    softmax = load_inline(
        name="softmax",
        cpp_sources=softmax_cpp_source,
        cuda_sources=softmax_source,
        functions=["log_softmax_cuda", "exp_cuda", "sum_cuda"],
        verbose=True,
        extra_cflags=[""],
        extra_ldflags=[""],
    )
except Exception as e:
    print(f"Error compiling CUDA module: {e}")
    raise

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super(ModelNew, self).__init__()
        self.log_softmax = softmax.log_softmax_cuda
        self.exp = softmax.exp_cuda
        self.sum = softmax.sum_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = self.log_softmax(x)
        probs = self.exp(log_probs)
        sums = self.sum(probs)
        return probs / sums.view(-1, 1)