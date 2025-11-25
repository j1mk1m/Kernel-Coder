import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_tanh_scale_add(
    const float* input,
    float scaling_factor,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * height * width)
        return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (channels * width * height);
    
    float val = input[idx];
    val = tanhf(val);
    val *= scaling_factor;
    val += bias[c];
    output[idx] = val;
}

torch::Tensor fused_tanh_scale_add_cuda(
    torch::Tensor input,
    float scaling_factor,
    torch::Tensor bias) {
    auto device = input.device();
    TORCH_CHECK(bias.device() == device, "Bias must be on the same device as input");
    
    auto output = torch::empty_like(input);
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int total_elements = batch_size * channels * height * width;
    
    const int block_size = 256;
    dim3 block(block_size);
    dim3 grid((total_elements + block_size - 1) / block_size);
    
    fused_tanh_scale_add<<<grid, block>>>(
        input.data_ptr<float>(),
        scaling_factor,
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, height, width
    );
    
    return output;
}
"""

elementwise_fused_cpp_source = (
    "torch::Tensor fused_tanh_scale_add_cuda(torch::Tensor input, float scaling_factor, torch::Tensor bias);"
)

elementwise_fused = load_inline(
    name="elementwise_fused",
    cpp_sources=elementwise_fused_cpp_source,
    cuda_sources=elementwise_fused_source,
    functions=["fused_tanh_scale_add_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = elementwise_fused.fused_tanh_scale_add_cuda(x, self.scaling_factor, self.bias)
        x = self.max_pool(x)
        return x