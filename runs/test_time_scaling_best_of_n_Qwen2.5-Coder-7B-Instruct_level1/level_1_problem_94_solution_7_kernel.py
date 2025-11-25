import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for mean squared error
mean_squared_error_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* predictions, const float* targets, float* mse, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mse[idx] = (predictions[idx] - targets[idx]) * (predictions[idx] - targets[idx]);
    }
}

__global__ void sum_kernel(const float* mse, float* result, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < size) ? mse[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

float mean_squared_error_cuda(const float* predictions, const float* targets, int size) {
    float mse[size];
    memset(mse, 0, sizeof(mse));

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mse_kernel<<<num_blocks, block_size>>>(predictions, targets, mse, size);

    float result = 0;
    sum_kernel<<<1, block_size, block_size * sizeof(float)>>>(mse, &result, size);

    return result / size;
}
"""

mean_squared_error_cpp_source = (
    "float mean_squared_error_cuda(const float* predictions, const float* targets, int size);"
)

# Compile the inline CUDA code for mean squared error
mean_squared_error = load_inline(
    name="mean_squared_error",
    cpp_sources=mean_squared_error_cpp_source,
    cuda_sources=mean_squared_error_source,
    functions=["mean_squared_error_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mean_squared_error = mean_squared_error

    def forward(self, predictions, targets):
        return self.mean_squared_error.mean_squared_error_cuda(predictions.cpu().numpy(), targets.cpu().numpy(), predictions.numel())