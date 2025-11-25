import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused ReLU + HardSwish CUDA kernel
fused_relu_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_relu_hardswish_kernel(const float* in, float* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        float x_relu = fmaxf(x, 0.0f);
        float scale = (x_relu + 3.0f) / 6.0f;
        scale = fminf(scale, 1.0f);
        out[idx] = x_relu * scale;
    }
}

torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input) {
    auto output = at::empty_like(input);
    int64_t size = input.numel();

    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_relu_hardswish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

fused_relu_hardswish_cpp = """
torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input);
"""

# Compile the fused activation kernel
fused_relu_hardswish = load_inline(
    name="fused_relu_hardswish",
    cpp_sources=fused_relu_hardswish_cpp,
    cuda_sources=fused_relu_hardswish_source,
    functions=["fused_relu_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_relu_hardswish = fused_relu_hardswish

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_relu_hardswish.fused_relu_hardswish_cuda(x)
        return x

# Keep these functions unchanged as per requirements
def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]