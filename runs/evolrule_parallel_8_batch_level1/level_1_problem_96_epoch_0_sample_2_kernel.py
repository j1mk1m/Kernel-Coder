import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 32768
input_shape = (32768,)
dim = 1

# Define the custom CUDA kernels for smooth L1 loss
elementwise_smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_loss_kernel(float* predictions, float* targets, float* block_sums, int N) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    float loss = 0.0f;
    if (idx < N) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabs(diff);
        loss = (abs_diff < 1.0f) ? 0.5f * diff * diff : (abs_diff - 0.5f);
    }
    shared[tid] = loss;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[bid] = shared[0];
    }
}

__global__ void reduce_kernel(float* input, float* output, int size) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    float value = 0.0f;
    if (idx < size) {
        value = input[idx];
    }
    shared[tid] = value;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[bid] = shared[0];
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto N = predictions.numel();
    assert(predictions.numel() == targets.numel());

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    auto block_sums_size = grid_size;
    auto block_sums = torch::empty({block_sums_size}, predictions.options());

    // Launch compute_loss_kernel
    compute_loss_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        N);
    cudaDeviceSynchronize();

    // Now reduce block_sums to get total sum
    auto current_size = block_sums_size;
    auto current_data = block_sums;

    while (current_size > 1) {
        int new_grid_size = (current_size + block_size - 1) / block_size;
        auto new_data = torch::empty({new_grid_size}, predictions.options());

        reduce_kernel<<<new_grid_size, block_size, block_size * sizeof(float)>>>(
            current_data.data_ptr<float>(),
            new_data.data_ptr<float>(),
            current_size);
        cudaDeviceSynchronize();

        current_data = new_data;
        current_size = new_grid_size;
    }

    float total_loss = current_data.item<float>();
    float mean_loss = total_loss / static_cast<float>(N);

    return torch::tensor({mean_loss}, predictions.options());
}
"""

elementwise_smooth_l1_loss_cpp_source = (
    "torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code for smooth L1 loss
smooth_l1_loss = load_inline(
    name="smooth_l1_loss",
    cpp_sources=elementwise_smooth_l1_loss_cpp_source,
    cuda_sources=elementwise_smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1_loss = smooth_l1_loss

    def forward(self, predictions, targets):
        return self.smooth_l1_loss.smooth_l1_loss_cuda(predictions, targets)

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape) * scale, torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []