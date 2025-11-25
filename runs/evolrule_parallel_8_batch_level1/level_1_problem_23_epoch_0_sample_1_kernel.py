import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int features) {
    extern __shared__ float shared_data[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    float local_max = -FLT_MAX;

    // Compute max for each thread's chunk
    for (int i = tid; i < features; i += blockDim.x) {
        float val = input[batch_idx * features + i];
        if (val > local_max) {
            local_max = val;
        }
    }

    // Write to shared memory for max reduction
    shared_data[threadIdx.x] = local_max;
    __syncthreads();

    // Reduce to find max_val
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_data[threadIdx.x] < shared_data[threadIdx.x + s]) {
                shared_data[threadIdx.x] = shared_data[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    float max_val = shared_data[0];
    __syncthreads();

    // Reset shared memory for sum calculation
    shared_data[threadIdx.x] = 0.0f;
    __syncthreads();

    // Compute exponents and accumulate sum
    for (int i = tid; i < features; i += blockDim.x) {
        float exp_val = expf(input[batch_idx * features + i] - max_val);
        shared_data[threadIdx.x] += exp_val;
    }

    __syncthreads();

    // Reduce to find sum_exp
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum_exp = shared_data[0];
    __syncthreads();

    // Compute output values
    for (int i = tid; i < features; i += blockDim.x) {
        float exp_val = expf(input[batch_idx * features + i] - max_val);
        output[batch_idx * features + i] = exp_val / sum_exp;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    if (!input.is_contiguous()) {
        AT_ERROR("Input must be contiguous");
    }
    if (!input.device().is_cuda()) {
        AT_ERROR("Input must be on CUDA");
    }

    const int batch_size = input.size(0);
    const int features = input.size(1);
    const int threads_per_block = 256;
    const int blocks = batch_size;

    auto output = torch::empty_like(input);

    int shared_mem_size = threads_per_block * sizeof(float);

    softmax_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features
    );

    return output;
}
"""

softmax_cpp_source = """
torch::Tensor softmax_cuda(torch::Tensor input);
"""

softmax = load_inline(
    name="softmax_cuda",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_cuda = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda.softmax_cuda(x)