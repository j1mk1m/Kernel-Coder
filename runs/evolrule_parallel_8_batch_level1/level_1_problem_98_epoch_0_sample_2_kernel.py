import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void kl_div_partial_sums(const float* predictions, const float* targets, float* partial_sums, int N) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float sum = 0.0f;
    for (int idx = bid * blockDim.x + tid; idx < N; idx += blockDim.x * gridDim.x) {
        float pred = predictions[idx];
        float target = targets[idx];

        if (target > 0 && pred > 0) {
            float log_t = logf(target);
            float log_p = logf(pred);
            float term = target * (log_t - log_p);
            sum += term;
        }
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

__global__ void final_reduce(float* partial_sums, int num_blocks, float* output) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;

    if (tid < num_blocks) {
        shared_mem[tid] = partial_sums[tid];
    } else {
        shared_mem[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_mem[0];
    }
}

torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets) {
    predictions = predictions.contiguous();
    targets = targets.contiguous();
    assert(predictions.device() == targets.device());

    int64_t batch_size = predictions.size(0);
    int64_t D = predictions.size(1);
    int N = batch_size * D;

    const int block_size = 256;
    const int grid_size = std::min(1024, (N + block_size - 1) / block_size);

    auto partial_sums = torch::empty({grid_size}, torch::CUDA(predictions.device().index()));

    kl_div_partial_sums<<<grid_size, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        N
    );

    const int final_block_size = std::max(grid_size, block_size);
    final_block_size = std::min(final_block_size, 1024);

    auto output = torch::empty(1, torch::CUDA(predictions.device().index()));
    float* output_ptr = output.data_ptr<float>();

    final_reduce<<<1, final_block_size, final_block_size * sizeof(float)>>>(
        partial_sums.data_ptr<float>(),
        grid_size,
        output_ptr
    );

    cudaDeviceSynchronize();

    *output_ptr = *output_ptr / static_cast<float>(batch_size);

    return output;
}
"""

kl_div_cpp_source = (
    "torch::Tensor kl_div_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

kl_div = load_inline(
    name="kl_div",
    cpp_sources=kl_div_cpp_source,
    cuda_sources=kl_div_source,
    functions=["kl_div_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return kl_div.kl_div_cuda(predictions, targets)