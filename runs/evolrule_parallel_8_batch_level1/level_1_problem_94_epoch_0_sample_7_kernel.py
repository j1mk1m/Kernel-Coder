import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MSE loss
elementwise_mse_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_reduction(const float* a, const float* b, float* sum, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    sdata[tid] = 0.0f;

    __syncthreads();

    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        float diff = a[i] - b[i];
        sdata[tid] += diff * diff;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

at::Tensor mse_loss_cuda(const at::Tensor& a, const at::Tensor& b) {
    AT_ASSERT(a.device().is_cuda() && b.device().is_cuda());
    AT_ASSERT(a.sizes() == b.sizes());

    const int64_t size = a.numel();
    const int threadsPerBlock = 256;
    const int blocksPerGrid = 1024;

    auto sum_dev = at::empty({1}, a.options());
    sum_dev.zero_();

    mse_reduction<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        sum_dev.data_ptr<float>(),
        size
    );

    AT_CUDA_CHECK(cudaGetLastError());

    auto mean = sum_dev / static_cast<float>(size);
    return mean;
}
"""

elementwise_mse_cpp_source = """
torch::Tensor mse_loss_cuda(const torch::Tensor& a, const torch::Tensor& b);
"""

# Compile the inline CUDA code
elementwise_mse = load_inline(
    name="elementwise_mse",
    cpp_sources=elementwise_mse_cpp_source,
    cuda_sources=elementwise_mse_source,
    functions=["mse_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_cuda = elementwise_mse

    def forward(self, predictions, targets):
        return self.mse_loss_cuda.mse_loss_cuda(predictions, targets)