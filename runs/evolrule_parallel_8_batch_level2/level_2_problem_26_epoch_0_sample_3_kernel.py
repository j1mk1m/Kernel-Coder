import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for addition and HardSwish
fused_add_hswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_hswish_kernel(
    const float* x_data, const float* add_data, float* out_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float temp = x_data[idx] + add_data[idx];
        float s_plus_3 = temp + 3.0f;
        float relu6_val = fmaxf(fminf(s_plus_3, 6.0f), 0.0f);
        float hardswish_val = temp * (relu6_val / 6.0f);
        out_data[idx] = temp * hardswish_val;
    }
}

torch::Tensor fused_add_hswish_cuda(
    torch::Tensor x, torch::Tensor add_input) {
    auto output = torch::empty_like(x);
    int size = x.numel();
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    fused_add_hswish_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), add_input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

fused_add_hswish_cpp = (
    "torch::Tensor fused_add_hswish_cuda(torch::Tensor x, torch::Tensor add_input);"
)

# Compile the fused CUDA kernel
fused_add_hswish = load_inline(
    name="fused_add_hswish",
    cuda_sources=fused_add_hswish_source,
    cpp_sources=fused_add_hswish_cpp,
    functions=["fused_add_hswish_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))  # Matches original structure
        self.fused_add_hswish = fused_add_hswish

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        return self.fused_add_hswish.fused_add_hswish_cuda(x, add_input)

# Compatibility with the original code's get_init_inputs
ModelNew.get_init_inputs = lambda: [32, 64, 3, 2, 1, 1, (64, 1, 1, 1, 1)]