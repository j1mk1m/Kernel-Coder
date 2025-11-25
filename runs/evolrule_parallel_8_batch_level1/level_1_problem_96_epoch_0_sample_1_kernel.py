import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

smooth_l1_loss_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_loss_kernel(
    const float* predictions,
    const float* targets,
    float* partial_sums,
    int N
) {
    extern __shared__ float block_sums[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float loss_val = 0.0f;

    int global_idx = bid * blockDim.x + tid;
    if (global_idx < N) {
        float delta = predictions[global_idx] - targets[global_idx];
        float abs_delta = fabsf(delta);
        if (abs_delta >= 1.0f) {
            loss_val = (abs_delta - 0.5f);
        } else {
            loss_val = 0.5f * delta * delta;
        }
    }

    block_sums[tid] = loss_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_sums[tid] += block_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[bid] = block_sums[0];
    }
}

torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int N = predictions.numel();
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;

    torch::Tensor partial_sums = torch::empty({num_blocks}, torch::dtype(torch::kFloat32).device(predictions.device()));
    
    smooth_l1_loss_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        N
    );

    float total = partial_sums.sum().item<float>();
    float loss = total / N;

    return torch::tensor({loss}, torch::dtype(torch::kFloat32).device(predictions.device()));
}
"""

smooth_l1_loss_cpp_source = """
torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code for smooth L1 loss
smooth_l1_loss = load_inline(
    name="smooth_l1_loss",
    cpp_sources=smooth_l1_loss_cpp_source,
    cuda_sources=smooth_l1_loss_cuda_source,
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