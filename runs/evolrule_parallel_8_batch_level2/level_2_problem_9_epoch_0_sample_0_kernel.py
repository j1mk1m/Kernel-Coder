import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise kernel (subtract, multiply, ReLU)
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise(
    const float* input, float* output,
    float subtract_val, float multiply_val,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = input[idx] - subtract_val;
        temp *= multiply_val;
        output[idx] = fmaxf(temp, 0.0f); // ReLU
    }
}

torch::Tensor fused_elementwise_cuda(torch::Tensor input,
                                    float subtract_val,
                                    float multiply_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

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

fused_elementwise_cpp = "torch::Tensor fused_elementwise_cuda(torch::Tensor input, float subtract_val, float multiply_val);"

# Compile the fused element-wise kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_elementwise_cpp,
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fused_elementwise = fused_elementwise
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_elementwise.fused_elementwise_cuda(
            x, self.subtract_value, self.multiply_value
        )
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]