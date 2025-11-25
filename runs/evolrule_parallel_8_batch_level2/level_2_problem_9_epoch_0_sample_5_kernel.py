import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise CUDA kernel (subtract, multiply, ReLU)
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise(
    const float* input,
    float* output,
    float subtract_val,
    float multiply_val,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] - subtract_val;
        val *= multiply_val;
        output[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

torch::Tensor fused_elementwise_cuda(torch::Tensor input, float subtract_val, float multiply_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    fused_elementwise<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        subtract_val,
        multiply_val,
        size
    );
    
    return output;
}
"""

fused_elementwise_cpp = """
torch::Tensor fused_elementwise_cuda(torch::Tensor input, float subtract_val, float multiply_val);
"""

# Compile the fused kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=[fused_elementwise_cpp],
    cuda_sources=[fused_elementwise_source],
    functions=["fused_elementwise_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_elementwise_fcn = fused_elementwise

    def forward(self, x):
        x = self.linear(x)
        return self.fused_elementwise_fcn.fused_elementwise_cuda(
            x, self.subtract_value, self.multiply_value
        )