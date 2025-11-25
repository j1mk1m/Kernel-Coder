import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_max_pool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void custom_max_pool2d_kernel(const float* input, float* output,
                                        int batch_size, int channels,
                                        int in_height, int in_width,
                                        int out_height, int out_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_height * out_width) return;

    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int n = idx / (channels * out_width * out_height);

    int h_in = h_out * 2;
    int w_in = w_out * 2;

    float max_val = -std::numeric_limits<float>::infinity();
    for (int kh = 0; kh < 2; ++kh) {
        for (int kw = 0; kw < 2; ++kw) {
            int h = h_in + kh;
            int w = w_in + kw;
            if (h < in_height && w < in_width) {
                float val = input[n * channels * in_height * in_width +
                                 c * in_height * in_width +
                                 h * in_width + w];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    output[idx] = max_val;
}

torch::Tensor custom_max_pool2d_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    auto out_height = (in_height + 1) / 2;
    auto out_width = (in_width + 1) / 2;

    auto output = torch::empty({batch_size, channels, out_height, out_width}, input.options());

    const int block_size = 256;
    int total_threads = batch_size * channels * out_height * out_width;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    custom_max_pool2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, in_height, in_width,
        out_height, out_width
    );

    return output;
}
"""

custom_max_pool2d_cpp_source = (
    "torch::Tensor custom_max_pool2d_cuda(torch::Tensor input);"
)

custom_max_pool2d = load_inline(
    name="custom_max_pool2d",
    cpp_sources=custom_max_pool2d_cpp_source,
    cuda_sources=custom_max_pool2d_source,
    functions=["custom_max_pool2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.tanh(x)
        x = custom_max_pool2d.custom_max_pool2d_cuda(x)
        x = self.group_norm(x)
        return x

batch_size = 512
in_channels = 64  
out_channels = 128  
height = width = 32
kernel_size = 5
stride = 1  
padding = 1
groups = 8
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]