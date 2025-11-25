import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU and bias addition CUDA kernel
fused_relu_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_add_bias(const float* input, const float* bias, float* output,
                                   int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * channels * height * width) {
        int c = (idx / (height * width)) % channels;
        float val = input[idx];
        val = fmaxf(val, 0.0f);
        val += bias[c];
        output[idx] = val;
    }
}

torch::Tensor fused_relu_add_cuda(torch::Tensor input, torch::Tensor bias) {
    int batch = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    // Check bias dimensions
    TORCH_CHECK(bias.sizes() == torch::IntArrayRef({channels, 1, 1}),
                "Bias tensor must be of shape (", channels, ", 1, 1)");
    
    auto output = torch::empty_like(input);
    
    int num_elements = batch * channels * height * width;
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    
    fused_relu_add_bias<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, channels, height, width);
    
    return output;
}
"""

# Compile the CUDA code inline
fused_relu_add = load_inline(
    name='fused_relu_add',
    cpp_sources=["torch::Tensor fused_relu_add_cuda(torch::Tensor input, torch::Tensor bias);"],
    cuda_sources=[fused_relu_add_source],
    functions=['fused_relu_add_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_relu_add = fused_relu_add  # Reference to the CUDA module

    def forward(self, x):
        x = self.conv(x)
        # Apply fused ReLU and bias addition
        x = self.fused_relu_add.fused_relu_add_cuda(x, self.bias)
        return x