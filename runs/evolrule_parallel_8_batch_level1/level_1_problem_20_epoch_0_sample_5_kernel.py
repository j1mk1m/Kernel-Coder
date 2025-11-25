import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Leaky ReLU
leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // Not needed here but included in case of future optimizations
#include <cmath>

template <typename scalar_t>
__global__ void leaky_relu_kernel(const scalar_t* __restrict__ input,
                                 scalar_t* __restrict__ output,
                                 const float negative_slope,
                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = input[idx];
        output[idx] = val > 0 ? val : val * static_cast<scalar_t>(negative_slope);
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope) {
    auto output = torch::empty_like(input);
    int elements = input.numel();
    int threads = 256;
    int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "leaky_relu_cuda", ([&] {
        leaky_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            negative_slope,
            elements);
    }));

    return output;
}
"""

# Compile the CUDA code
leaky_relu_cpp_source = "torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope);"
leaky_relu = load_inline(
    name="leaky_relu",
    cpp_sources=leaky_relu_cpp_source,
    cuda_sources=leaky_relu_source,
    functions=["leaky_relu_cuda"],
    verbose=False,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-arch=sm_70"],  # Adjust based on CUDA architecture
)

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, &gt;init__):
            self.negative_slope = negative_slope
            self.custom_leaky_relu = leaky_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_leaky_relu.leaky_relu_cuda(x, self.negative_slope)

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []