import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Global parameters as per the original code
batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

# Define the fused kernel source
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise_kernel(
    const float* input,
    float subtract_val,
    float multiply_val,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] - subtract_val;
        val *= multiply_val;
        output[idx] = fmaxf(val, 0.0f);
    }
}

torch::Tensor fused_elementwise_cuda(torch::Tensor input, float subtract_val, float multiply_val) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        subtract_val,
        multiply_val,
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

fused_elementwise_header = """
torch::Tensor fused_elementwise_cuda(torch::Tensor input, float subtract_val, float multiply_val);
"""

# Compile the fused kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_elementwise_header,
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_elementwise = fused_elementwise  # Store the loaded module

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_elementwise.fused_elementwise_cuda(
            x, self.subtract_value, self.multiply_value
        )
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]