import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for online softmax
softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* a, float* b, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        sdata[tid] = exp(a[i]);
        __syncthreads();

        int s = blockDim.x / 2;
        while (s > 0) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
            s >>= 1;
        }

        if (tid == 0) {
            sdata[0] = 1.0f / sdata[0];
        }
        __syncthreads();

        if (i < size) {
            b[i] = sdata[tid] * exp(a[i]);
        }
    }
}

torch::Tensor softmax_cuda(torch::Tensor a) {
    auto size = a.numel();
    auto b = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softmax_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(a.data_ptr<float>(), b.data_ptr<float>(), size);

    return b;
}
"""

softmax_cpp_source = (
    "torch::Tensor softmax_cuda(torch::Tensor a);"
)

# Compile the inline CUDA code for online softmax
softmax = load_inline(
    name="softmax",
    cpp_sources=softmax_cpp_source,
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.softmax = softmax

    def forward(self, x):
        x = self.matmul(x)
        x = self.add_value + x
        x = self.softmax.softmax_cuda(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]