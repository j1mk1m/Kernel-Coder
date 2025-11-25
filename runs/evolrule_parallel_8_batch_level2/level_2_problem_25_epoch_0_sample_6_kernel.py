import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

min_tanh_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define FLT_MAX 3.402823466e+38F

__global__ void min_tanh_tanh_kernel(
    const float* input, float* output,
    int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int b = idx / (H * W);

    float min_val = FLT_MAX;
    for (int c = 0; c < C; ++c) {
        int offset = b * C * H * W + c * H * W + h * W + w;
        float val = input[offset];
        if (val < min_val) {
            min_val = val;
        }
    }

    float val = tanhf(tanhf(min_val));
    output[b * H * W + h * W + w] = val;
}

torch::Tensor min_tanh_tanh_cuda(torch::Tensor input) {
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty({B, 1, H, W}, input.options());

    const int threads_per_block = 256;
    const int blocks = (B * H * W + threads_per_block - 1) / threads_per_block;

    min_tanh_tanh_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W);

    return output;
}
"""

min_tanh_tanh_cpp_source = """
torch::Tensor min_tanh_tanh_cuda(torch::Tensor input);
"""

min_tanh_tanh = load_inline(
    name="min_tanh_tanh",
    cpp_sources=min_tanh_tanh_cpp_source,
    cuda_sources=min_tanh_tanh_source,
    functions=["min_tanh_tanh_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).cuda()
        self.min_tanh_tanh = min_tanh_tanh

    def forward(self, x):
        x = x.cuda()
        x = self.conv(x).contiguous()
        x = self.min_tanh_tanh.min_tanh_tanh_cuda(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]