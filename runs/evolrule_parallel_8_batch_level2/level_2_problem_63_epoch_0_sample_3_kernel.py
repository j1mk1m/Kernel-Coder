import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU and division CUDA kernel
relu_divide_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void fused_relu_divide_kernel(const float* input, float* output, int size, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = fmaxf(val, 0.0f) / divisor;
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
        size,
        divisor
    );

    return output;
}
"""

relu_divide_cpp_source = """
torch::Tensor fused_relu_divide_cuda(torch::Tensor input, float divisor);
"""

# Compile the inline CUDA code
relu_divide = load_inline(
    name="relu_divide",
    cpp_sources=relu_divide_cpp_source,
    cuda_sources=relu_divide_source,
    functions=["fused_relu_divide_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)
        self.divisor = divisor
        self.relu_divide = relu_divide  # Load the fused kernel

    def forward(self, x):
        # Compute matmul + bias using addmm (optimized)
        intermediate = torch.addmm(self.bias, x, self.weight.t())
        # Apply fused ReLU and division via custom kernel
        out = self.relu_divide.fused_relu_divide_cuda(intermediate, self.divisor)
        return out

def get_inputs():
    # Assuming batch_size, in_features, etc. are defined elsewhere
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, divisor]