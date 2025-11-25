import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_row_sums(const float* predictions, const float* targets, float* row_sums, int N, int D) {
    int i = blockIdx.x;
    float y_i = targets[i];
    float sum_row = 0.0f;

    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        int idx = i * D + j;
        float p = predictions[idx];
        float value = 1.0f - p * y_i;
        sum_row += fmaxf(value, 0.0f);
    }

    extern __shared__ float shared[];
    int tid = threadIdx.x;
    shared[tid] = sum_row;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_sums[i] = shared[0];
    }
}

__global__ void sum_rows(const float* row_sums, float* total_sum, int N, int D) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        sum += row_sums[i];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *total_sum = shared[0] / (N * D);
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int N = predictions.size(0);
    const int D = predictions.size(1);

    auto row_sums = torch::empty({N}, predictions.options());
    auto total_sum = torch::empty({1}, predictions.options());

    const int block_size_row = 256;
    dim3 grid_row(N);
    dim3 block_row(block_size_row);

    compute_row_sums<<<grid_row, block_row, block_row.x * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        row_sums.data_ptr<float>(),
        N, D
    );

    const int block_size_sum = 256;
    dim3 grid_sum(1);
    dim3 block_sum(block_size_sum);

    sum_rows<<<grid_sum, block_sum, block_sum.x * sizeof(float)>>>(
        row_sums.data_ptr<float>(),
        total_sum.data_ptr<float>(),
        N, D
    );

    return total_sum;
}
"""

hinge_loss_cpp_source = """
torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

# Compile the inline CUDA code
hinge_loss = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hinge_loss_cuda = hinge_loss

    def forward(self, predictions, targets):
        return self.hinge_loss_cuda.hinge_loss_cuda(predictions, targets)