import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for MSE loss computation
mse_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_reduction_kernel(const float* a, const float* b, float* partial_sums, int N) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float sum = 0.0f;

    for (int i = bid * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    shared_mem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[bid] = shared_mem[0];
    }
}

torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int N = predictions.numel();
    const int block_size = 1024;
    const int num_blocks = (N + block_size - 1) / block_size;

    auto partial_sums = torch::empty({num_blocks}, torch::Device(torch::kCUDA).dtype(torch::kFloat32));
    auto partial_sums_data = partial_sums.data_ptr<float>();

    mse_reduction_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums_data,
        N
    );

    // Copy partial_sums to host and compute total_sum
    float* h_partial_sums = new float[num_blocks];
    cudaMemcpy(h_partial_sums, partial_sums_data, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total_sum = 0.0f;
    for (int i = 0; i < num_blocks; ++i) {
        total_sum += h_partial_sums[i];
    }

    delete[] h_partial_sums;

    return torch::tensor({total_sum / N}, predictions.options());
}
"""

mse_loss_cpp_source = (
    "torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code for MSE loss
mse_loss = load_inline(
    name="mse_loss",
    cpp_sources=mse_loss_cpp_source,
    cuda_sources=mse_loss_source,
    functions=["mse_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = mse_loss

    def forward(self, predictions, targets):
        return self.mse_loss.mse_loss_cuda(predictions, targets)