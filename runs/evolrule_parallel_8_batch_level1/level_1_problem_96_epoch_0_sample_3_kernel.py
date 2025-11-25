import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for smooth L1 loss
smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_forward_kernel(
    const float* predictions,
    const float* targets,
    float* loss_sum,
    int total_elements,
    float beta
) {
    extern __shared__ float partial_sums[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    float sum = 0.0f;

    if (idx < total_elements) {
        float x = predictions[idx] - targets[idx];
        float abs_x = fabsf(x);
        if (abs_x <= beta) {
            sum += 0.5f * (x * x) / beta;
        } else {
            sum += abs_x - 0.5f * beta;
        }
    }

    partial_sums[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss_sum, partial_sums[0]);
    }
}

torch::Tensor smooth_l1_loss_forward_cuda(
    torch::Tensor predictions,
    torch::Tensor targets,
    float beta = 1.0f
) {
    auto device = predictions.device();
    auto options = predictions.options();

    const int total_elements = predictions.numel();
    const int block_size = 1024;  // Larger block size for better performance
    const int grid_size = (total_elements + block_size - 1) / block_size;

    auto loss_sum = torch::empty({1}, options);

    // Calculate shared memory size as block_size * sizeof(float)
    const size_t shared_mem_size = block_size * sizeof(float);

    smooth_l1_loss_forward_kernel<<<grid_size, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        loss_sum.data_ptr<float>(),
        total_elements,
        beta
    );

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    float mean_loss = loss_sum.item<float>() / total_elements;

    return torch::full({1}, mean_loss, options);
}
"""

# Compile the inline CUDA code for Smooth L1 Loss
smooth_l1_loss = load_inline(
    name="smooth_l1_loss",
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = 1.0  # Default beta value for Smooth L1 Loss
        self.smooth_l1_loss = smooth_l1_loss  # Use the custom CUDA implementation

    def forward(self, predictions, targets):
        return self.smooth_l1_loss.smooth_l1_loss_forward_cuda(
            predictions, targets, self.beta
        ).squeeze()