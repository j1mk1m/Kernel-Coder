import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

mse_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void mse_kernel(const T* predictions, const T* targets, T* total_sum, int num_elements) {
    extern __shared__ T shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    T local_sum = 0.0;

    if (idx < num_elements) {
        T diff = predictions[idx] - targets[idx];
        local_sum = diff * diff;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total_sum, shared[0]);
    }
}

at::Tensor mse_loss_cuda(at::Tensor predictions, at::Tensor targets) {
    const int block_size = 1024;
    int num_elements = predictions.numel();
    int grid_size = (num_elements + block_size - 1) / block_size;

    auto total_sum = at::empty({1}, predictions.options()).zero_();
    auto total_sum_ptr = total_sum.data_ptr<float>();

    mse_kernel<float><<<grid_size, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        total_sum_ptr,
        num_elements
    );

    cudaDeviceSynchronize();

    auto mean = total_sum / static_cast<float>(num_elements);
    return mean;
}
"""

mse_loss_cpp_source = """
at::Tensor mse_loss_cuda(at::Tensor predictions, at::Tensor targets);
"""

mse_loss = load_inline(
    name="mse_loss",
    cpp_sources=mse_loss_cpp_source,
    cuda_sources=mse_loss_source,
    functions=["mse_loss_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = mse_loss

    def forward(self, predictions, targets):
        return self.mse_loss.mse_loss_cuda(predictions, targets)