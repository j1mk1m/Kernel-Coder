import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused kernel for min, tanh, and another tanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_min_tanh_tanh_kernel(
    const float* input, float* output,
    int batch_size, int out_channels, int height, int width,
    int channels_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * height * width) return;

    // Compute min along channels dimension (dim=1)
    float min_val = input[idx * out_channels];
    for (int c = 1; c < channels_dim; ++c) {
        min_val = fminf(min_val, input[idx * out_channels + c]);
    }

    // Apply tanh twice
    float val = tanhf(tanhf(min_val));

    output[idx] = val;
}

torch::Tensor fused_min_tanh_tanh_cuda(
    torch::Tensor input,
    int channels_dim) {

    int batch_size = input.size(0);
    int height = input.size(2);
    int width = input.size(3);
    auto output = torch::empty({batch_size, 1, height, width}, input.options());

    const int elements = batch_size * height * width;
    const int threads = 256;
    const int blocks = (elements + threads - 1) / threads;

    fused_min_tanh_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input.size(1),
        height,
        width,
        channels_dim);

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = "torch::Tensor fused_min_tanh_tanh_cuda(torch::Tensor input, int channels_dim);"

# Compile the fused kernel
fused_kernel = load_inline(
    name="fused_min_tanh_tanh",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_min_tanh_tanh_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_min_tanh_tanh = fused_kernel

    def forward(self, x):
        x = self.conv(x)
        # Get channels dimension for kernel (dim=1 after conv)
        channels_dim = x.size(1)
        x = self.fused_min_tanh_tanh.fused_min_tanh_tanh_cuda(x, channels_dim)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 16
    height = width = 256
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [16, 64, 3]