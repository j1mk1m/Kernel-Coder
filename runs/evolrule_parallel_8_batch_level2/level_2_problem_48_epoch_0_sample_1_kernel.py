import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_operations_kernel(
    const float* input, const float* scaling, const float* bias, float* output,
    int batch_size, int channels, int depth, int height, int width) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * depth * height * width;

    if (index < total_size) {
        // Compute channel index
        int elements_per_channel = depth * height * width;
        int c = (index / elements_per_channel) % channels;

        float val = input[index] * scaling[c];
        val = tanhf(val);
        val *= bias[c];
        val = 1.0f / (1.0f + expf(-val));

        output[index] = val;
    }
}

torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor scaling, torch::Tensor bias) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    int num_elements = batch_size * channels * depth * height * width;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), scaling.data_ptr<float>(),
        bias.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, depth, height, width
    );

    cudaDeviceSynchronize(); // Ensure kernel completion

    return output;
}
"""

fused_ops_cpp_source = (
    "torch::Tensor fused_operations_cuda(torch::Tensor input, torch::Tensor scaling, torch::Tensor bias);"
)

# Compile the fused operations CUDA kernel
fused_operations = load_inline(
    name="fused_operations",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape).cuda())
        self.bias = nn.Parameter(torch.randn(bias_shape).cuda())
        self.fused_operations = fused_operations  # The loaded CUDA module

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_operations.fused_operations_cuda(x, self.scaling_factor, self.bias)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 3
    depth, height, width = 16, 64, 64
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    scaling_factor = 2
    bias_shape = (out_channels, 1, 1, 1)
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]