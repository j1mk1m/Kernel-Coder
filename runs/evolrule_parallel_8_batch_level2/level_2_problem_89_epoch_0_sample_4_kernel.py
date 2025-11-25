import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused kernel for Subtract, Swish, and Channel Max
fused_operations_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void fused_sub_swish_max_kernel(
    const scalar_t* input_data,
    const scalar_t* sub_params,
    scalar_t* output_data,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * depth * height * width) return;

    scalar_t max_val = -INFINITY;

    for (int c = 0; c < channels; ++c) {
        int input_offset = c * batch_size * depth * height * width + idx;
        scalar_t val = input_data[input_offset] - sub_params[c];

        // Compute Swish: x * sigmoid(x)
        val = val / (1 + exp(-val));  // Sigmoid
        val = val * val;  // Swish = x * sigmoid(x)

        if (val > max_val) {
            max_val = val;
        }
    }

    output_data[idx] = max_val;
}

torch::Tensor fused_sub_swish_max_cuda(
    torch::Tensor input,
    torch::Tensor sub_params) {

    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    auto output = torch::empty({batch_size, depth, height, width}, input.options());

    int total_elements = batch_size * depth * height * width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_sub_swish_max_cuda", ([&] {
        fused_sub_swish_max_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            sub_params.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size, channels, depth, height, width);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_sub_swish_max_cuda", &fused_sub_swish_max_cuda, "Fused subtract-Swish-Max kernel");
}
"""

# Compile the fused kernel
fused_ops = load_inline(
    name="fused_operations",
    cpp_sources=[""],
    cuda_sources=fused_operations_kernel,
    functions=["fused_sub_swish_max_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))  # Subtract parameters
        self.fused_ops = fused_ops  # Fused kernel

    def forward(self, x):
        # Convolution and Max Pool remain as standard PyTorch ops
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        
        # Softmax (kept as PyTorch op for numerical stability)
        x = torch.softmax(x, dim=1)
        
        # Apply fused kernel for subtract, Swish, and channel-wise max
        x = self.fused_ops.fused_sub_swish_max_cuda(x, self.subtract.view(1, -1, 1, 1, 1))
        
        return x

# Ensure compatibility with original input functions
def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding]