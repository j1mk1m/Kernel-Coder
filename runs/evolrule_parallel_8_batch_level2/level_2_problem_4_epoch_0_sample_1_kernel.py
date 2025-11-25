import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Mish CUDA kernel
fused_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float compute_mish(float x) {
    float abs_x = fabsf(x);
    float sign_x = (x > 0) ? 1.0f : -1.0f;
    float max_term = x * 0.5f * (1.0f + sign_x); // Equivalent to max(x, 0)
    float exp_term = expf(-abs_x);
    float log_term = logf(1.0f + exp_term);
    float softplus_x = max_term + log_term;
    float tanh_softplus_x = tanhf(softplus_x);
    return x * tanh_softplus_x;
}

__global__ void fused_mish_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float y = compute_mish(x);
        output[idx] = compute_mish(y);
    }
}

torch::Tensor fused_mish_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_mish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

fused_mish_cpp_source = (
    "torch::Tensor fused_mish_cuda(torch::Tensor input);"
)

# Compile the fused Mish kernel
fused_mish = load_inline(
    name="fused_mish",
    cpp_sources=fused_mish_cpp_source,
    cuda_sources=fused_mish_source,
    functions=["fused_mish_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_mish = fused_mish  # Stores the compiled kernel module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_mish.fused_mish_cuda(x)  # Invoke fused Mish kernel
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]