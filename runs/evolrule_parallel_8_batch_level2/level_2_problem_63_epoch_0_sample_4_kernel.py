import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused ReLU + division CUDA kernel
fused_relu_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_relu_divide_kernel(const float* input, float* output, float divisor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        val = fmaxf(val, 0.0f);
        val /= divisor;
        output[idx] = val;
    }
}

torch::Tensor fused_relu_divide_cuda(torch::Tensor input, float divisor) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_relu_divide_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        size
    );

    return output;
}
"""

fused_relu_divide_header = """
torch::Tensor fused_relu_divide_cuda(torch::Tensor input, float divisor);
"""

# Compile fused kernel
fused_relu_divide = load_inline(
    name="fused_relu_divide",
    cpp_sources=fused_relu_divide_header,
    cuda_sources=fused_relu_divide_source,
    functions=["fused_relu_divide_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        self.fused_relu_divide = fused_relu_divide

    def forward(self, x):
        x = self.linear(x)
        x = self.fused_relu_divide.fused_relu_divide_cuda(x, self.divisor)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, divisor]