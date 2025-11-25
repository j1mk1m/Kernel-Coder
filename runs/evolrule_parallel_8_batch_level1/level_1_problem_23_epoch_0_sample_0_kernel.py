import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void custom_softmax(float* input, float* output, int batch_size, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];

    // Step 1: Compute max
    float local_max = -FLT_MAX;
    int start = tid * (dim / blockDim.x);
    int end = (tid + 1) * (dim / blockDim.x);
    if (tid == blockDim.x - 1) end = dim;

    for (int i = start; i < end; ++i) {
        float val = input[row * dim + i];
        if (val > local_max) local_max = val;
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float global_max = sdata[0];
    __syncthreads();

    // Step 2: Compute sum of exp(x_i - global_max)
    float local_sum = 0.0f;
    for (int i = start; i < end; ++i) {
        float val = input[row * dim + i];
        float exp_val = expf(val - global_max);
        local_sum += exp_val;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float total_sum = sdata[0];
    __syncthreads();

    // Step 3: Compute final output
    for (int i = start; i < end; ++i) {
        float val = input[row * dim + i];
        float exp_val = expf(val - global_max);
        float result = exp_val / total_sum;
        output[row * dim + i] = result;
    }
}

extern "C" {
    void custom_softmax_cuda(torch::Tensor input, torch::Tensor output, int batch_size, int dim) {
        int block_size = 1024;
        int num_blocks = batch_size;
        int shared_mem = 2 * block_size * sizeof(float);
        custom_softmax<<<num_blocks, block_size, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);
    }
}
"""

# Compile the inline CUDA code
softmax_cuda = load_inline(
    name="softmax_cuda",
    cuda_sources=softmax_cuda_source,
    functions=["custom_softmax_cuda"],
    verbose=True,
    extra_cuda_cflags=["-lineinfo", "-std=c++14"],
    extra_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_cuda = softmax_cuda

    def forward(self, x: torch.Tensor):
        batch_size, dim = x.size()
        output = torch.empty_like(x)
        self.softmax_cuda.custom_softmax_cuda(x, output, batch_size, dim)
        return output