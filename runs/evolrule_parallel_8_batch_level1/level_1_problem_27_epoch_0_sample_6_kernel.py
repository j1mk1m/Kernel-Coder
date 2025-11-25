import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# SELU parameters as per PyTorch's implementation
LAMBDA = 1.0507
ALPHA = 1.6732632423543772848170429916717
SCALE = 1.0507009873554804934193349852946

# CUDA kernel for SELU activation
selu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void selu_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float x = input[tid];
        output[tid] = x > 0.0f ? LAMBDA * x : LAMBDA * ALPHA * (expf(x) - 1.0f);
    }
}

torch::Tensor selu_cuda(torch::Tensor input) {
    int n = input.numel();
    torch::Tensor output = torch::empty({n}, torch::dtype(input.dtype()).device(input.device()));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    selu_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
    cudaDeviceSynchronize();

    return output.view_as(input);
}
"""

# Compile the inline CUDA code for SELU
selu_cpp_source = "torch::Tensor selu_cuda(torch::Tensor input);"
selu_extension = load_inline(
    name="selu_ext",
    cpp_sources=selu_cpp_source,
    cuda_sources=selu_source,
    functions=["selu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.selu_cuda = selu_extension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.selu_cuda.selu_cuda(x)