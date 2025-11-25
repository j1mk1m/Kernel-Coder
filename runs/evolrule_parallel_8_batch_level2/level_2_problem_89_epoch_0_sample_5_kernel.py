import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom Subtract + Swish kernel
subtract_swish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void subtract_swish_kernel(const float* input, const float* sub_params, float* output, int batch, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * channels * depth * height * width)
        return;

    int w = idx % width;
    idx /= width;
    int h = idx % height;
    idx /= height;
    int d = idx % depth;
    idx /= depth;
    int c = idx % channels;
    int b = idx / channels;

    float val = input[idx] - sub_params[c];
    val = val / (1.0f + expf(-val)) * val; // Swish activation
    output[idx] = val;
}

torch::Tensor subtract_swish_cuda(torch::Tensor input, torch::Tensor sub_params) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    int num_elements = batch * channels * depth * height * width;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "subtract_swish_cuda", ([&] {
        subtract_swish_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            sub_params.data_ptr<float>(),
            output.data_ptr<float>(),
            batch, channels, depth, height, width
        );
    }));

    return output;
}
"""

subtract_swish_cpp_source = (
    "torch::Tensor subtract_swish_cuda(torch::Tensor input, torch::Tensor sub_params);"
)

subtract_swish = load_inline(
    name="subtract_swish",
    cpp_sources=subtract_swish_cpp_source,
    cuda_sources=subtract_swish_source,
    functions=["subtract_swish_cuda"],
    verbose=True,
)

# Custom Channel Max kernel
channel_max_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void channel_max_kernel(const float* input, float* output, int batch, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * depth * height * width)
        return;

    int w = idx % width;
    idx /= width;
    int h = idx % height;
    idx /= height;
    int d = idx % depth;
    int b = idx / depth;

    float max_val = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
        int in_idx = b * channels * depth * height * width + c * depth * height * width + d * height * width + h * width + w;
        float val = input[in_idx];
        if (val > max_val)
            max_val = val;
    }
    output[idx] = max_val;
}

torch::Tensor channel_max_cuda(torch::Tensor input) {
    auto batch = input.size(0);
    auto channels = input.size(1);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto output = torch::zeros({batch, depth, height, width}, input.options());

    const int block_size = 256;
    int num_elements = batch * depth * height * width;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "channel_max_cuda", ([&] {
        channel_max_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch, channels, depth, height, width
        );
    }));

    return output;
}
"""

channel_max_cpp_source = (
    "torch::Tensor channel_max_cuda(torch::Tensor input);"
)

channel_max = load_inline(
    name="channel_max",
    cpp_sources=channel_max_cpp_source,
    cuda_sources=channel_max_source,
    functions=["channel_max_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))  # Subtract parameter
        self.subtract_swish = subtract_swish
        self.channel_max = channel_max

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = torch.softmax(x, dim=1)
        # Apply custom subtract + swish
        x = self.subtract_swish.subtract_swish_cuda(x, self.subtract.view(-1))
        # Apply custom channel max
        x = self.channel_max.channel_max_cuda(x)
        return x

# Keep the input generation functions as before
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding]