import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for element-wise scaling by a scalar
scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_kernel(const float* in, float scale, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] * scale;
    }
}

torch::Tensor scale_cuda(torch::Tensor in, float scale) {
    auto size = in.numel();
    auto out = torch::empty_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scale_kernel<<<num_blocks, block_size>>>(
        in.data_ptr<float>(),
        scale,
        out.data_ptr<float>(),
        size
    );

    return out;
}
"""

scale_cpp = "torch::Tensor scale_cuda(torch::Tensor in, float scale);"

# Custom fused CUDA kernel for adding bias and scaling by a scalar
fused_add_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_scale_kernel(
    const float* x,
    const float* bias,
    float scale,
    float* out,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * depth * height * width) return;

    int c = (idx / (depth * height * width)) % channels;
    float bias_val = bias[c];
    out[idx] = (x[idx] + bias_val) * scale;
}

torch::Tensor fused_add_scale_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    float scale
) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto depth = x.size(2);
    auto height = x.size(3);
    auto width = x.size(4);

    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_elements = batch_size * channels * depth * height * width;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_add_scale_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale,
        out.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width
    );

    return out;
}
"""

fused_add_scale_cpp = "torch::Tensor fused_add_scale_cuda(torch::Tensor x, torch::Tensor bias, float scale);"

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

        # Load custom CUDA kernels
        self.scale1_kernel = load_inline(
            name="scale",
            cuda_sources=scale_source,
            cpp_sources=scale_cpp,
            functions=["scale_cuda"],
            verbose=True,
        )

        self.fused_add_scale = load_inline(
            name="fused_add_scale",
            cuda_sources=fused_add_scale_source,
            cpp_sources=fused_add_scale_cpp,
            functions=["fused_add_scale_cuda"],
            verbose=True,
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        # Apply scale1 using custom kernel
        x = self.scale1_kernel.scale_cuda(x, self.scale1.item())
        x = self.avg_pool(x)
        # Apply fused add bias and scale2
        x = self.fused_add_scale.fused_add_scale_cuda(x, self.bias, self.scale2.item())
        return x