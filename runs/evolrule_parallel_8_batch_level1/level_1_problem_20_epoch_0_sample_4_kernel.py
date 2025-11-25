import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code for LeakyReLU
leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float value = input[idx];
        output[idx] = (value > 0.0f) ? value : value * negative_slope;
    }
}

torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    leaky_relu_kernel<<<num_blocks, block_size, 0, stream>>>(input.data_ptr<float>(), output.data_ptr<float>(), negative_slope, size);

    return output;
}
"""

# Compile the CUDA kernel
leaky_relu_cuda = load_inline(
    name="leaky_relu_cuda",
    cpp_sources="",
    cuda_sources=leaky_relu_source,
    functions=["leaky_relu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
        self.leaky_relu_cuda = leaky_relu_cuda  # Bind the CUDA kernel to the model

    def forward(self, x):
        return self.leaky_relu_cuda.leaky_relu_cuda(x, self.negative_slope)

# Keep the input generation functions same as original for compatibility
def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed