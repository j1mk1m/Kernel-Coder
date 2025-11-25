import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cpp_source = """
torch::Tensor fused_div_leaky_relu_cuda(torch::Tensor input, float divisor, float negative_slope);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_div_leaky_relu(const float* input, float* output, float divisor, float negative_slope, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx] / divisor;
        output[idx] = (x < 0) ? x * negative_slope : x;
    }
}

torch::Tensor fused_div_leaky_relu_cuda(torch::Tensor input, float divisor, float negative_slope) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    fused_div_leaky_relu<<<num_blocks, block_size>>>(input.data_ptr<float>(), 
                                                     output.data_ptr<float>(), 
                                                     divisor, 
                                                     negative_slope, 
                                                     size);
    return output;
}
"""

fused_div_leaky = load_inline(
    name="fused_div_leaky",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_div_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.fused_div_leaky = fused_div_leaky

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_div_leaky.fused_div_leaky_relu_cuda(x, self.divisor, 0.01)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]