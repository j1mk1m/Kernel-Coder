import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Smooth L1 Loss
smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void smooth_l1_loss_kernel(const float* predictions, const float* targets, float* block_sums, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int idx = bid * blockDim.x + tid;

    float loss = 0.0f;

    if (idx < N) {
        float diff = predictions[idx] - targets[idx];
        float abs_diff = fabs(diff);
        loss = (abs_diff < 1.0f) ? 0.5f * abs_diff * abs_diff : (abs_diff - 0.5f);
    }

    sdata[tid] = loss;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[bid] = sdata[0];
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int block_size = 1024;
    int N = predictions.numel();
    int num_blocks = (N + block_size - 1) / block_size;

    auto block_sums = torch::empty({num_blocks}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    smooth_l1_loss_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        N
    );

    float total_loss = block_sums.sum().item<float>();
    float mean_loss = total_loss / static_cast<float>(N);

    return torch::tensor({mean_loss}, torch::device(torch::kCUDA));
}
"""

smooth_l1_loss_cpp_source = (
    "torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code for Smooth L1 Loss
smooth_l1_loss = load_inline(
    name="smooth_l1_loss",
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1_loss_cuda = smooth_l1_loss

    def forward(self, predictions, targets):
        return self.smooth_l1_loss_cuda.smooth_l1_loss_cuda(predictions, targets)