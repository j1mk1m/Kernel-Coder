import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LogSumExp
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void logsumexp_kernel(const float* x, float* out, int batch_size, int dim_size) {
    extern __shared__ float sdata[];
    int batch_idx = blockIdx.x;

    float max_val = -INFINITY;
    float sum = 0.0;

    // Compute max for each thread's portion
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = x[batch_idx * dim_size + i];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Store max in shared memory
    sdata[threadIdx.x] = max_val;
    __syncthreads();

    // Reduce to find the block's max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x] < sdata[threadIdx.x + s]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float block_max = sdata[0]; // The maximum for the row

    // Compute sum of exp(x_i - block_max)
    sum = 0.0;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float term = expf(x[batch_idx * dim_size + i] - block_max);
        sum += term;
    }

    // Store sum in shared memory
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduce the sums
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[batch_idx] = logf(sdata[0]) + block_max;
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim_size = x.size(1);
    auto out = torch::empty({batch_size}, x.options());

    const int block_size = 256;
    int shared_mem = block_size * sizeof(float);

    logsumexp_kernel<<<batch_size, block_size, shared_mem>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim_size
    );

    return out;
}
"""

logsumexp_cpp_source = "torch::Tensor logsumexp_cuda(torch::Tensor x);"

# Compile the LogSumExp CUDA kernel
logsumexp = load_inline(
    name="logsumexp",
    cpp_sources=logsumexp_cpp_source,
    cuda_sources=logsumexp_source,
    functions=["logsumexp_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.logsumexp = logsumexp  # The custom CUDA function

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = self.logsumexp.logsumexp_cuda(x)  # Use custom kernel
        return x

batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, output_size]