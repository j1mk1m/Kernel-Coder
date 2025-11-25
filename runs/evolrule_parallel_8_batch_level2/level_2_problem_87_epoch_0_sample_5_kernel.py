import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for subtract and Mish activation
mish_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void mish_subtract_kernel(const float* input, float* output, float subtract_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] - subtract_val;

        float abs_x = fabsf(x);
        float exp_term = expf(-abs_x);
        float softplus_val;

        if (x > 0) {
            softplus_val = x + log1pf(exp_term);
        } else {
            softplus_val = log1pf(exp_term);
        }

        float tanh_softplus = tanhf(softplus_val);
        output[idx] = x * tanh_softplus;
    }
}

torch::Tensor mish_subtract_cuda(torch::Tensor input, float subtract_val) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mish_subtract_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), subtract_val, size
    );

    return output;
}
"""

mish_subtract_cpp_source = "torch::Tensor mish_subtract_cuda(torch::Tensor input, float subtract_val);"

# Compile the kernel once outside the class
mish_subtract = load_inline(
    name="mish_subtract",
    cpp_sources=mish_subtract_cpp_source,
    cuda_sources=mish_subtract_source,
    functions=["mish_subtract_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.mish_subtract = mish_subtract  # Reference the pre-compiled kernel

    def forward(self, x):
        x = self.conv(x)
        total_subtract = self.subtract_value_1 + self.subtract_value_2
        x = self.mish_subtract.mish_subtract_cuda(x, total_subtract)
        return x