import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused add and HardSwish
fused_add_hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_hardswish_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = a[idx] + b[idx];
        float h = val + 3;
        h = (h < 0) ? 0 : (h > 6) ? 6 : h;
        h /= 6;
        out[idx] = val * h;
    }
}

torch::Tensor fused_add_hardswish_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::empty_like(a); // Use empty_like for efficiency

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_add_hardswish_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

fused_add_hardswish_cpp_source = (
    "torch::Tensor fused_add_hardswish_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code
fused_add_hardswish = load_inline(
    name="fused_add_hardswish",
    cpp_sources=fused_add_hardswish_cpp_source,
    cuda_sources=fused_add_hardswish_source,
    functions=["fused_add_hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))  # Retained for compatibility
        self.fused_add_hardswish = fused_add_hardswish

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        # Ensure inputs are contiguous for CUDA kernel
        x_contig = x.contiguous()
        add_contig = add_input.contiguous()
        x = self.fused_add_hardswish.fused_add_hardswish_cuda(x_contig, add_contig)
        return x