import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused LeakyReLU + Add + Clamp CUDA kernel
fused_leaky_add_clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_leaky_add_clamp_kernel(
    const float* in_data,
    const float* sum_tensor_data,
    float* out_data,
    int batch_size,
    int out_channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width)
        return;

    int w = idx % width;
    int h = (idx / width) % height;
    int d = (idx / (width * height)) % depth;
    int c = (idx / (width * height * depth)) % out_channels;
    int n = idx / (out_channels * depth * height * width);

    float x = in_data[idx];
    float leaky = (x > 0.0f) ? x : 0.2f * x;
    leaky += sum_tensor_data[c];
    float clamped = fmaxf(fminf(leaky, 1.0f), -1.0f);
    out_data[idx] = clamped;
}

torch::Tensor fused_leaky_add_clamp_cuda(torch::Tensor in, torch::Tensor sum_tensor) {
    auto batch_size = in.size(0);
    auto out_channels = in.size(1);
    auto depth = in.size(2);
    auto height = in.size(3);
    auto width = in.size(4);
    auto size = in.numel();
    auto out = torch::empty_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_leaky_add_clamp_kernel<<<num_blocks, block_size>>>(
        in.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
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

fused_leaky_add_clamp_cpp_source = (
    "torch::Tensor fused_leaky_add_clamp_cuda(torch::Tensor in, torch::Tensor sum_tensor);"
)

# GELU approximation CUDA kernel
gelu_approx_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gelu_approx_kernel(const float* in_data, float* out_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = in_data[idx];
    float x_cubed = x * x * x;
    float temp = 0.7978845608f * (x + 0.044715f * x_cubed);
    float tanh_temp = tanhf(temp);
    out_data[idx] = 0.5f * x * (1.0f + tanh_temp);
}

torch::Tensor gelu_approx_cuda(torch::Tensor in) {
    auto size = in.numel();
    auto out = torch::empty_like(in);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_approx_kernel<<<num_blocks, block_size>>>(in.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

gelu_approx_cpp_source = "torch::Tensor gelu_approx_cuda(torch::Tensor in);"

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        
        # Load fused kernel
        self.fused_leaky_add_clamp = load_inline(
            name="fused_leaky_add_clamp",
            cpp_sources=fused_leaky_add_clamp_cpp_source,
            cuda_sources=fused_leaky_add_clamp_source,
            functions=["fused_leaky_add_clamp_cuda"],
            verbose=True
        )
        
        # Load GELU approximation kernel
        self.gelu_approx = load_inline(
            name="gelu_approx",
            cpp_sources=gelu_approx_cpp_source,
            cuda_sources=gelu_approx_source,
            functions=["gelu_approx_cuda"],
            verbose=True
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_leaky_add_clamp.fused_leaky_add_clamp_cuda(x, self.sum_tensor)
        x = self.gelu_approx.gelu_approx_cuda(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]