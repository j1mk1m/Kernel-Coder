import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise operations (add bias, multiply scale, apply sigmoid) CUDA kernel
fused_elementwise_sigmoid_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void fused_elementwise_sigmoid_kernel(
    const float* input, const float* bias, const float* scale, float* output,
    int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels * height * width) return;

    // Compute 4D indices
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (channels * width * height);

    float val = input[idx];
    val += bias[c]; // bias is (channels, 1, 1), so use channel index
    val *= scale[c]; // same for scale
    output[idx] = 1.0f / (1.0f + expf(-val));
}

torch::Tensor fused_elementwise_sigmoid_cuda(
    torch::Tensor input, torch::Tensor bias, torch::Tensor scale) {
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    CHECK_INPUT(scale);

    auto output = torch::empty_like(input);

    int batch = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int total_elements = batch * channels * height * width;

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_elementwise_sigmoid_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, height, width
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\\n", cudaGetErrorString(err));

    return output;
}
"""

# Header for the fused element-wise CUDA function
fused_elementwise_sigmoid_header = """
torch::Tensor fused_elementwise_sigmoid_cuda(torch::Tensor input, torch::Tensor bias, torch::Tensor scale);
"""

# Compile the CUDA kernel
fused_elementwise_sigmoid = load_inline(
    name="fused_elementwise_sigmoid",
    cpp_sources=fused_elementwise_sigmoid_header,
    cuda_sources=fused_elementwise_sigmoid_source,
    functions=["fused_elementwise_sigmoid_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.fused_elementwise_sigmoid = fused_elementwise_sigmoid  # Attach CUDA module to model

    def forward(self, x):
        x = self.conv(x)
        # Apply fused element-wise operations via CUDA kernel
        x = self.fused_elementwise_sigmoid.fused_elementwise_sigmoid_cuda(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x

# Configuration and input generation (unchanged from original)
batch_size = 128
in_channels = 8
out_channels = 32
height = width = 256
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    # Inputs moved to CUDA as required by custom kernels
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]