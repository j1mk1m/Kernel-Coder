import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise CUDA kernel
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_elementwise(const float* original_x, const float* bias, float* output, int total_elements, int spatial_size, int num_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int c = (idx / spatial_size) % num_channels;
    float C = original_x[idx];
    float b = bias[c]; 

    float temp1 = C + b;
    float temp2 = temp1 + C;
    float temp3 = temp2 * C;
    float temp4 = temp3 + C;

    output[idx] = temp4;
}

torch::Tensor fused_elementwise_cuda(torch::Tensor original_x, torch::Tensor bias) {
    auto total_elements = original_x.numel();
    auto num_channels = original_x.size(1);
    auto D = original_x.size(2);
    auto H = original_x.size(3);
    auto W = original_x.size(4);
    auto spatial_size = D * H * W;

    auto output = torch::zeros_like(original_x);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_elementwise<<<num_blocks, block_size>>>(
        original_x.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        spatial_size,
        num_channels
    );

    return output;
}
"""

fused_elementwise_cpp_header = (
    "torch::Tensor fused_elementwise_cuda(torch::Tensor original_x, torch::Tensor bias);"
)

# Compile the fused CUDA code
fused_elementwise = load_inline(
    name="fused_elementwise",
    cpp_sources=fused_elementwise_cpp_header,
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_elementwise = fused_elementwise

    def forward(self, x):
        conv_out = self.conv_transpose(x)
        x = self.fused_elementwise.fused_elementwise_cuda(conv_out, self.bias)
        return x

def get_inputs():
    batch_size = 16
    in_channels = 32
    depth, height, width = 16, 32, 32
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1
    bias_shape = (out_channels, 1, 1, 1)
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]