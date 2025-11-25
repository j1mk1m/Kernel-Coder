import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_logsumexp_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_logsumexp_relu_kernel(
    const float* input, float* output,
    int batch_size, int in_channels, 
    int out_depth, int out_height, int out_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_depth * out_height * out_width) return;

    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int b = idx / (out_depth * out_height * out_width);

    float max_val = -INFINITY;
    for (int c = 0; c < in_channels; ++c) {
        int in_idx = b * in_channels * out_depth * out_height * out_width
                    + c * out_depth * out_height * out_width
                    + d * out_height * out_width
                    + h * out_width
                    + w;
        float val = input[in_idx];
        if (val > max_val) {
            max_val = val;
        }
    }

    float sum = 0.0f;
    for (int c = 0; c < in_channels; ++c) {
        int in_idx = b * in_channels * out_depth * out_height * out_width
                    + c * out_depth * out_height * out_width
                    + d * out_height * out_width
                    + h * out_width
                    + w;
        float val = input[in_idx];
        sum += expf(val - max_val);
    }

    float log_sum_exp = max_val + logf(sum);
    float result = log_sum_exp > 0.0f ? log_sum_exp : 0.0f;

    int out_idx = b * out_depth * out_height * out_width
                 + 0 * out_depth * out_height * out_width
                 + d * out_height * out_width
                 + h * out_width
                 + w;
    output[out_idx] = result;
}

torch::Tensor fused_logsumexp_relu_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_depth = input.size(2);
    int out_height = input.size(3);
    int out_width = input.size(4);

    auto output = torch::zeros({batch_size, 1, out_depth, out_height, out_width}, input.options());

    int numel = batch_size * out_depth * out_height * out_width;
    const int threads_per_block = 256;
    const int blocks_per_grid = (numel + threads_per_block - 1) / threads_per_block;

    fused_logsumexp_relu_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_depth,
        out_height,
        out_width
    );

    return output;
}
"""

fused_logsumexp_relu_cpp = (
    "torch::Tensor fused_logsumexp_relu_cuda(torch::Tensor input);"
)

fused_logsumexp_relu = load_inline(
    name="fused_logsumexp_relu",
    cpp_sources=fused_logsumexp_relu_cpp,
    cuda_sources=fused_logsumexp_relu_source,
    functions=["fused_logsumexp_relu_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fused_logsumexp_relu = fused_logsumexp_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.fused_logsumexp_relu.fused_logsumexp_relu_cuda(x)
        return x

batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]