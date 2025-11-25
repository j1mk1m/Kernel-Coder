import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise CUDA kernel (subtract1, tanh, subtract2)
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise_kernel(const float* in_data, float* out_data, 
                                         const float a, const float b, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = in_data[idx] - a;
        val = tanhf(val);  // Apply tanh
        val -= b;
        out_data[idx] = val;
    }
}

torch::Tensor fused_elementwise_cuda(torch::Tensor input, float a, float b) {
    auto output = torch::empty_like(input);
    const int num_elements = input.numel();
    const int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    
    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        a, 
        b, 
        num_elements
    );
    
    return output;
}
"""

fused_elementwise_cpp_source = (
    "torch::Tensor fused_elementwise_cuda(torch::Tensor input, float a, float b);"
)

# Compile the fused element-wise CUDA kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_elementwise_cpp_source,
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.fused_elementwise = fused_elementwise  # Load the fused CUDA kernel

    def forward(self, x):
        x = self.conv(x)
        # Apply fused element-wise operations (subtract1 + tanh + subtract2)
        x = self.fused_elementwise.fused_elementwise_cuda(
            x, self.subtract1_value, self.subtract2_value
        )
        x = self.avgpool(x)
        return x

# Compatibility with original get_inputs and get_init_inputs functions
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]