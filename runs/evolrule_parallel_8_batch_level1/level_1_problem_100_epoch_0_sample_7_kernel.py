import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_hinge_loss_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_partial_sums(const float* predictions, const float* targets, float* partial_sums, int B, int N) {
    extern __shared__ float shared_mem[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_idx = threadIdx.x;
    float val = 0.0f;

    if (tid < B * N) {
        int i = tid / N;
        int j = tid % N;
        float pred_val = predictions[i * N + j];
        float target_val = targets[i];
        val = 1.0f - pred_val * target_val;
        val = fmaxf(val, 0.0f);
    }

    shared_mem[thread_idx] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            shared_mem[thread_idx] += shared_mem[thread_idx + s];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        partial_sums[blockIdx.x] = shared_mem[0];
    }
}

__global__ void reduce_partial_sums(const float* partial_sums, int num_partial_sums, float* total_sum) {
    extern __shared__ float shared_mem[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_idx = threadIdx.x;
    float val = 0.0f;

    if (tid < num_partial_sums) {
        val = partial_sums[tid];
    }

    shared_mem[thread_idx] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            shared_mem[thread_idx] += shared_mem[thread_idx + s];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        atomicAdd(total_sum, shared_mem[0]);
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int B = predictions.size(0);
    int N = predictions.size(1);
    int total_elements = B * N;

    const int block_size = 1024;
    int num_blocks_first = (total_elements + block_size - 1) / block_size;

    auto partial_sums = torch::empty({num_blocks_first}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    compute_partial_sums<<<num_blocks_first, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        B, N
    );

    int num_partial_sums = num_blocks_first;
    int num_blocks_second = (num_partial_sums + block_size - 1) / block_size;

    auto total_sum = torch::zeros(1, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    reduce_partial_sums<<<num_blocks_second, block_size, block_size * sizeof(float)>>>(
        partial_sums.data_ptr<float>(),
        num_partial_sums,
        total_sum.data_ptr<float>()
    );

    float mean = total_sum.item<float>() / static_cast<float>(total_elements);
    return torch::tensor({mean}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
}
"""

elementwise_hinge_loss_cpp = """
torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

hinge_loss_cuda = load_inline(
    name="hinge_loss_cuda",
    cpp_sources=elementwise_hinge_loss_cpp,
    cuda_sources=elementwise_hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hinge_loss_cuda = hinge_loss_cuda

    def forward(self, predictions, targets):
        return self.hinge_loss_cuda.hinge_loss_cuda(predictions, targets)