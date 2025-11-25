import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA kernel for fused GELU with constant memory optimization
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Use constant memory for precomputed constants
__constant__ float SQRT_2_OVER_PI;
__constant__ float COEF;

// Initialize constants at runtime
extern "C" {
    __global__ void init_constants() {
        // Constants are initialized once on the first kernel call
        if (threadIdx.x == 0) {
            SQRT_2_OVER_PI = sqrt(2.0f / M_PI);
            COEF = 0.044715f;
        }
        __syncthreads();
    }
}

__global__ void fused_gelu_kernel(const float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float xi = x[idx];
        float x_cubed = xi * xi * xi;
        float inner = xi + COEF * x_cubed;
        float tanh_val = tanh(SQRT_2_OVER_PI * inner);
        y[idx] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor fused_gelu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    // Initialize constants (only once)
    static bool initialized = false;
    if (!initialized) {
        init_constants<<<1, 1>>>();
        initialized = true;
    }

    fused_gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), size);
    return y;
}
"""

gelu_cpp_header = "torch::Tensor fused_gelu_cuda(torch::Tensor x);"

# Compile the CUDA kernel
fused_gelu = load_inline(
    name="fused_gelu",
    cpp_sources=gelu_cpp_header,
    cuda_sources=gelu_source,
    functions=["fused_gelu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_op = fused_gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_op.fused_gelu_cuda(x)