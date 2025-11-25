import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Define the custom CUDA kernel for fused operations (sum over features)
        fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void reduceRows(const T* __restrict__ g_idata, T* __restrict__ g_odata, int batch_size, int features) {
    extern __shared__ T sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Initialize shared memory
    sdata[tid] = 0;
    for (int i = tid; i < features; i += blockDim.x) {
        sdata[tid] += g_idata[bid * features + i];
    }
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        // Warp-level reduction
        volatile T* vdata = sdata;
        vdata[tid] += vdata[tid + 32];
        vdata[tid] += vdata[tid + 16];
        vdata[tid] += vdata[tid + 8];
        vdata[tid] += vdata[tid + 4];
        vdata[tid] += vdata[tid + 2];
        vdata[tid] += vdata[tid + 1];
    }
    __syncthreads();

    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}

torch::Tensor fused_operations_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int features = input.size(1);
    auto output = torch::empty({batch_size, 1}, input.options());

    const int block_size = 256;
    int shared_size = block_size * sizeof(float);

    reduceRows<float><<<batch_size, block_size, shared_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), batch_size, features);

    return output;
}
"""
        fused_ops = load_inline(
            name="fused_ops",
            cpp_sources=["torch::Tensor fused_operations_cuda(torch::Tensor input);"],
            cuda_sources=[fused_ops_source],
            functions=["fused_operations_cuda"],
            verbose=True,
        )
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_ops.fused_operations_cuda(x).view(-1, 1)
        return x