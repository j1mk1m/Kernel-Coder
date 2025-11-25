import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define FLOAT4_SIZE 4

__global__ void softplus_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        float xi = x[i];
        if (xi > 20.0f) {
            out[i] = xi;
        } else if (xi < -20.0f) {
            out[i] = exp(xi);
        } else {
            out[i] = log(1.0f + exp(xi));
        }
    }
}

// Optimized version using float4 vectorization
__global__ void softplus_vector_kernel(const float* __restrict__ x, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x * FLOAT4_SIZE + threadIdx.x;
    float4 val = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 res = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = idx; i < size; i += blockDim.x * gridDim.x * FLOAT4_SIZE) {
        val = ((float4*)x)[i];
        for (int j = 0; j < 4; j++) {
            float xi = val[j];
            if (xi > 20.0f) {
                res[j] = xi;
            } else if (xi < -20.0f) {
                res[j] = exp(xi);
            } else {
                res[j] = log(1.0f + exp(xi));
            }
        }
        ((float4*)out)[i] = res;
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = THREADS_PER_BLOCK;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Decide which kernel to launch based on data alignment
    if (size % FLOAT4_SIZE == 0 && ((size_t)x.data_ptr() % 16) == 0) {
        softplus_vector_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    } else {
        softplus_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    }

    return out;
}
"""

# Compile the inline CUDA code
softplus_cpp_source = "torch::Tensor softplus_cuda(torch::Tensor x);"
softplus_extension = load_inline(
    name="softplus",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus_cuda = softplus_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus_cuda.softplus_cuda(x)

# Ensure inputs are on GPU for CUDA execution
def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]