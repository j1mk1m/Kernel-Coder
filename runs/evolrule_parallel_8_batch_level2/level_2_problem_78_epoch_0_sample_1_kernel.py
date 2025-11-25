import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pool3d_kernel(const float* input, float* output,
    int batch, int channels, int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch * channels * out_depth * out_height * out_width) {
        return;
    }

    int out_w = idx % out_width;
    int temp = idx / out_width;
    int out_h = temp % out_height;
    temp = temp / out_height;
    int out_d = temp % out_depth;
    temp = temp / out_depth;
    int channel = temp % channels;
    int batch_idx = temp / channels;

    int in_d_start = out_d * 2;
    int in_h_start = out_h * 2;
    int in_w_start = out_w * 2;

    float max_val = -INFINITY;

    for (int kd = 0; kd < 2; ++kd) {
        int id = in_d_start + kd;
        if (id >= in_depth) continue;
        for (int kh = 0; kh < 2; ++kh) {
            int ih = in_h_start + kh;
            if (ih >= in_height) continue;
            for (int kw = 0; kw < 2; ++kw) {
                int iw = in_w_start + kw;
                if (iw >= in_width) continue;

                int input_offset = batch_idx * channels * in_depth * in_height * in_width +
                                   channel * in_depth * in_height * in_width +
                                   id * in_height * in_width +
                                   ih * in_width +
                                   iw;
                float current_val = input[input_offset];
                if (current_val > max_val) {
                    max_val = current_val;
                }
            }
        }
    }

    int output_offset = batch_idx * channels * out_depth * out_height * out_width +
                        channel * out_depth * out_height * out_width +
                        out_d * out_height * out_width +
                        out_h * out_width +
                        out_w;
    output[output_offset] = max_val;
}

torch::Tensor max_pool3d_cuda(torch::Tensor input) {
    int batch = input.size(0);
    int channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_depth = (in_depth + 1) / 2;
    int out_height = (in_height + 1) / 2;
    int out_width = (in_width + 1) / 2;

    auto output = torch::empty({batch, channels, out_depth, out_height, out_width}, 
                              input.options());

    int total_elements = batch * channels * out_depth * out_height * out_width;
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    max_pool3d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, in_depth, in_height, in_width,
        out_depth, out_height, out_width);

    return output;
}
"""

max_pool3d_cpp_source = (
    "torch::Tensor max_pool3d_cuda(torch::Tensor input);"
)

max_pool3d = load_inline(
    name="max_pool3d",
    cpp_sources=max_pool3d_cpp_source,
    cuda_sources=max_pool3d_source,
    functions=["max_pool3d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = max_pool3d  # Replaced with custom kernel
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1.max_pool3d_cuda(x)  # Call custom kernel
        x = self.max_pool2(x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x