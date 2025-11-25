import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    int tid = threadIdx.x;

    __shared__ float shared_max[1024];
    __shared__ float shared_sum[1024];

    // Step 1: Compute the maximum value for the row
    float current_max = -FLT_MAX;
    for (int i = tid; i < dim; i += 1024) {
        float val = input[row * dim + i];
        if (val > current_max) {
            current_max = val;
        }
    }
    shared_max[tid] = current_max;
    __syncthreads();

    // Reduce to find row maximum
    for (int s = 512; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid] < shared_max[tid + s]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }
    float row_max = shared_max[0];
    __syncthreads();

    // Step 2: Compute exponentials and accumulate sum
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += 1024) {
        float val = input[row * dim + i] - row_max;
        float exp_val = expf(val);
        output[row * dim + i] = exp_val;
        local_sum += exp_val;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduce to find total sum
    for (int s = 512; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float total_sum = shared_sum[0];
    __syncthreads();

    // Step 3: Normalize by dividing by total_sum
    for (int i = tid; i < dim; i += 1024) {
        output[row * dim + i] /= total_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int dim = input.size(1);

    dim3 blocks(batch_size);
    dim3 threads(1024);

    softmax_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output;
}
"""

softmax_cpp_source = """
#include <torch/extension.h>

torch::Tensor softmax_cuda(torch::Tensor input);
"""

softmax_cuda_ext = load_inline(
    name="softmax_cuda",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_cuda_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_cuda_ext = softmax_cuda_ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda_ext.softmax_cuda(x)