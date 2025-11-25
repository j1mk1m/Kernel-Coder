import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom GELU kernel
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        const float kAlpha = 0.7978845608f; // sqrt(2/pi)
        float term = kAlpha * (x + 0.044715f * x * x * x);
        float tanh_term = tanhf(term);
        output[idx] = 0.5f * x * (1.0f + tanh_term);
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor input);"

# Compile the GELU kernel
gelu_mod = load_inline(
    name="gelu_mod",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups
        )
        self.group_norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels
        )
        self.gelu_mod = gelu_mod  # The GELU module

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.gelu_mod.gelu_cuda(x)
        x = self.group_norm(x)
        return x