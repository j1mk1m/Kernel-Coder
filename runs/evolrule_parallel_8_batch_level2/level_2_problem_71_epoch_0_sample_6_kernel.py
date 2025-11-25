import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for division and LeakyReLU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_div_leaky_relu_kernel(
    const float* input, float* output, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] / divisor;
        output[idx] = val > 0.0f ? val : val * 0.01f;
    }
}

torch::Tensor fused_div_leaky_relu_cuda(torch::Tensor input, float divisor) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_div_leaky_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        size
    );

    return output;
}
"""

fused_header = """
torch::Tensor fused_div_leaky_relu_cuda(torch::Tensor input, float divisor);
"""

# Load the fused CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cuda_sources=fused_kernel_source,
    cpp_sources=fused_header,
    functions=["fused_div_leaky_relu_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        x = self.conv(x)
        x = fused_ops.fused_div_leaky_relu_cuda(x, self.divisor)
        return x