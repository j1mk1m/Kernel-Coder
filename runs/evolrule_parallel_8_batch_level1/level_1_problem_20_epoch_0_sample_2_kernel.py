import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for LeakyReLU
leaky_relu_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    leaky_relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(),
                                                 output.data_ptr<float>(),
                                                 negative_slope,
                                                 size);
    return output;
}
"""

leaky_relu_cuda_header = """
torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope);
"""

# Compile the CUDA kernel
leaky_relu_cuda_mod = load_inline(
    name="leaky_relu_cuda",
    cpp_sources=leaky_relu_cuda_header,
    cuda_sources=leaky_relu_cuda_source,
    functions=["leaky_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
        self.leaky_relu_cuda_mod = leaky_relu_cuda_mod  # Store the compiled module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu_cuda_mod.leaky_relu_cuda(x, self.negative_slope)

batch_size = 4096
dim = 393216

def get_inputs():
    # Ensure inputs are on GPU for CUDA kernel compatibility
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []