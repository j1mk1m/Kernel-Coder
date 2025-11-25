import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel code
fused_operations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_operations_kernel(const float* input, const float* bias, float* output, int num_elements, int out_channels, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int batch = idx / (out_channels * D * H * W);
    int remainder = idx % (out_channels * D * H * W);
    int c = remainder / (D * H * W);

    float x = input[idx];

    // Apply ReLU
    x = fmaxf(x, 0.0f);

    // Apply Leaky ReLU (with slope 0.01)
    x = (x >= 0) ? x : 0.01f * x;

    // Apply GELU approximation
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    float tanh_inner = tanhf(inner);
    x = 0.5f * x * (1.0f + tanh_inner);

    // Apply Sigmoid
    float sigmoid_x = 1.0f / (1.0f + expf(-x));

    // Add bias
    float b = bias[c];
    output[idx] = sigmoid_x + b;
}

torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor bias) {
    // Check inputs are on the same device
    auto device = input.device();
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be on CUDA");
    TORCH_CHECK(input.device() == bias.device(), "Input and bias must be on the same device");

    int batch_size = input.size(0);
    int out_channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int num_elements = input.numel();

    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int grid_size = (num_elements + block_size - 1) / block_size;

    fused_operations_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        out_channels,
        D, H, W
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

fused_operations_cpp_source = "torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile the fused CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=[fused_operations_cpp_source],
    cuda_sources=[fused_operations_source],
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.fused_ops = fused_ops  # Store the CUDA module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_operations_cuda(x, self.bias)
        return x

def get_inputs():
    batch_size = 64
    in_channels = 8
    depth, height, width = 32, 64, 64
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 8
    out_channels = 32
    kernel_size = 3
    bias_shape = (out_channels, 1, 1, 1)
    return [in_channels, out_channels, kernel_size, bias_shape]