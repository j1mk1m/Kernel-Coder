import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for masked cumulative sum
masked_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void masked_cumsum_kernel(const float* x, const bool* mask, float* out, int batch_size, int input_size, int dim) {
    // Implement the masked cumulative sum using shared memory and coalesced memory access patterns
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x + tid;

    if (i < input_size) {
        sdata[tid] = mask[i] ? x[i] : 0.0f;
    } else {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&out[bid], sdata[0]);
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto out = torch::zeros(batch_size, dtype=torch.float32).to(device=x.device);

    const int block_size = 256;
    const int num_blocks = (input_size + block_size - 1) / block_size;

    masked_cumsum_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
        x.data_ptr<float>(), mask.data