import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for fused activation operations
fused_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_activation_kernel(
    const float* x,
    const float* sum_t,
    float* out,
    int batch_size,
    int out_channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size * out_channels * depth * height * width) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int d = (idx / (width * height)) % depth;
    int c = (idx / (width * height * depth)) % out_channels;
    int b = idx / (out_channels * depth * height * width);

    float val = x[idx];

    // Apply Leaky ReLU with slope 0.2
    if (val < 0) val *= 0.2;

    // Add the corresponding element from sum_tensor
    val += sum_t[c];

    // Clamp between -1 and 1
    if (val < -1.0) val = -1.0;
    else if (val > 1.0) val = 1.0;

    // Apply GELU approximation
    float x_gelu = val;
    float inner = sqrt(2.0f / M_PI) * (x_gelu + 0.044715f * x_gelu * x_gelu * x_gelu);
    float tanh_val = tanh(inner);
    val = 0.5f * x_gelu * (1.0f + tanh_val);

    out[idx] = val;
}

torch::Tensor fused_activation_cuda(
    torch::Tensor x,
    torch::Tensor sum_t
) {
    int batch_size = x.size(0);
    int out_channels = x.size(1);
    int depth = x.size(2);
    int height = x.size(3);
    int width = x.size(4);

    auto out = torch::empty_like(x);
    int total_elements = batch_size * out_channels * depth * height * width;

    const int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_activation_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        sum_t.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_channels,
        depth,
        height,
        width
    );

    return out;
}
"""

fused_activation_cpp_source = (
    "torch::Tensor fused_activation_cuda(torch::Tensor x, torch::Tensor sum_t);"
)

# Compile the fused activation CUDA code
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_activation_cpp_source,
    cuda_sources=fused_activation_source,
    functions=["fused_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_activation = fused_activation

    def forward(self, x):
        x = self.conv(x)
        return self.fused_activation.fused_activation_cuda(x, self.sum_tensor)