import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Constants for SELU as per PyTorch's implementation
alpha = 1.67326324235437728481607162126739417
scale = 1.05070098735548049341933498529469644

# CUDA kernel source for SELU
selu_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define the kernel function
template<typename scalar_t>
__global__ void selu_kernel(const scalar_t* __restrict__ input,
                           scalar_t* __restrict__ output,
                           const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        scalar_t x = input[idx];
        output[idx] = x > 0 ? scale * x : scale * alpha * (exp(x) - 1);
    }
}

// Wrapper function to launch the kernel
torch::Tensor selu_cuda(torch::Tensor input) {
    const int n = input.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    auto output = torch::empty_like(input);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel based on data type
    if (input.scalar_type() == torch::kFloat32) {
        selu_kernel<float><<<blocks, threads, 0, stream>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n);
    } else {
        AT_ERROR("Unsupported tensor type");
    }

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s", cudaGetErrorString(err));
    }

    return output;
}
"""

# Inline compile the CUDA code
selu_module = load_inline(
    name="custom_selu",
    cpp_sources="",
    cuda_sources=selu_cuda_source,
    functions=["selu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # Bind the CUDA function
        self.selu_cuda = selu_module.selu_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_cuda(x)

# Ensure inputs are on CUDA
def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []