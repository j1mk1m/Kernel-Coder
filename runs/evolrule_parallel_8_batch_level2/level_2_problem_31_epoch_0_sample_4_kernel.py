import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Global variables as in the original problem
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

# CUDA kernel code for fused_min_add_scale
elementwise_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void fused_min_add_scale_kernel(
    const float* x,
    float constant_value,
    const float* bias,
    float scaling_factor,
    float* out,
    int batch, int channels, int height, int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels * height * width)
        return;

    // Compute indices
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (width * height * channels);

    // Compute value
    float val = x[n * channels * height * width + c * height * width + h * width + w];
    val = fminf(val, constant_value);
    val += bias[c];
    val *= scaling_factor;

    // Write to output
    out[n * channels * height * width + c * height * width + h * width + w] = val;
}

torch::Tensor fused_min_add_scale_cuda(torch::Tensor x, float constant_value, torch::Tensor bias, float scaling_factor) {
    // Check if inputs are on the same device
    if (x.device() != bias.device()) {
        AT_ERROR("x and bias must be on the same device");
    }

    // Check for contiguous memory
    if (!x.is_contiguous()) {
        AT_ERROR("x must be contiguous");
    }
    if (!bias.is_contiguous()) {
        AT_ERROR("bias must be contiguous");
    }

    // Get dimensions
    int batch = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    // Output tensor
    auto out = torch::empty_like(x);

    // Number of elements
    int total_elements = batch * channels * height * width;

    // Block and grid dimensions
    const int block_size = 256;
    dim3 blocks((total_elements + block_size - 1) / block_size);
    dim3 threads(block_size);

    // Launch kernel
    fused_min_add_scale_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        constant_value,
        bias.data_ptr<float>(),
        scaling_factor,
        out.data_ptr<float>(),
        batch, channels, height, width
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return out;
}
"""

elementwise_fused_cpp = """
torch::Tensor fused_min_add_scale_cuda(torch::Tensor x, float constant_value, torch::Tensor bias, float scaling_factor);
"""

elementwise_fused = load_inline(
    name="elementwise_fused",
    cpp_sources=elementwise_fused_cpp,
    cuda_sources=elementwise_fused_source,
    functions=["fused_min_add_scale_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_flags=["-O3"],
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_min_add_scale = elementwise_fused  # The loaded extension module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_min_add_scale.fused_min_add_scale_cuda(
            x,
            self.constant_value,
            self.bias,
            self.scaling_factor,
        )
        return x

def get_inputs():
    # Generate input tensors on CUDA
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]