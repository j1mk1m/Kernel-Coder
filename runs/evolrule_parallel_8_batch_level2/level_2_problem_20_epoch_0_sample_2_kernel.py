import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_fusion_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_operations_kernel(
    const float* conv_out,
    const float* bias,
    float* out,
    int total_elements,
    int out_channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements)
        return;

    int c = (idx / (depth * height * width)) % out_channels;

    float conv_val = conv_out[idx];
    float bias_val = bias[c];

    float temp1 = conv_val + bias_val;
    float temp2 = temp1 + conv_val;
    float temp3 = temp2 * conv_val;
    float result = temp3 + conv_val;

    out[idx] = result;
}

torch::Tensor fused_operations_cuda(
    torch::Tensor conv_out,
    torch::Tensor bias,
    int total_elements,
    int out_channels,
    int depth,
    int height,
    int width
) {
    auto out = torch::empty_like(conv_out);
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_operations_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements,
        out_channels,
        depth,
        height,
        width
    );

    return out;
}
"""

elementwise_fusion_header = """
#include <torch/extension.h>
torch::Tensor fused_operations_cuda(
    torch::Tensor conv_out,
    torch::Tensor bias,
    int total_elements,
    int out_channels,
    int depth,
    int height,
    int width
);
"""

fusion_module = load_inline(
    name="fused_operations",
    cuda_sources=elementwise_fusion_source,
    cpp_sources=elementwise_fusion_header,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fusion = fusion_module  # Access the fused operation function

    def forward(self, x):
        conv_out = self.conv_transpose(x)
        # Get the dimensions from conv_out
        batch_size, out_channels, depth, height, width = conv_out.shape
        total_elements = conv_out.numel()
        # Call the fused CUDA kernel
        return self.fusion.fused_operations_cuda(
            conv_out, self.bias, total_elements, out_channels, depth, height, width
        )