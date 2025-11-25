import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise CUDA kernel
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise_kernel(
    const float* in_data,
    float* out_data,
    float a,
    float b,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = in_data[idx] - a;
        val = tanhf(val);
        val -= b;
        out_data[idx] = val;
    }
}

torch::Tensor fused_elementwise_cuda(torch::Tensor in, float a, float b) {
    auto size = in.numel();
    auto out = torch::empty_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        in.data_ptr<float>(),
        out.data_ptr<float>(),
        a,
        b,
        size
    );

    return out;
}
"""

fused_elementwise_cpp_source = (
    "torch::Tensor fused_elementwise_cuda(torch::Tensor in, float a, float b);"
)

# Compile the fused element-wise CUDA kernel
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_elementwise_cpp_source,
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.fused_elementwise = fused_elementwise

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_elementwise.fused_elementwise_cuda(
            x, self.subtract1_value, self.subtract2_value
        )
        x = self.avgpool(x)
        return x