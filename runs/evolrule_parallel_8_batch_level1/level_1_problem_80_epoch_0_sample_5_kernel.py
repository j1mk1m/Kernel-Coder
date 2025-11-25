import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input_data, const float* weight_data, float* output_data,
                             int batch_size, int in_channels, int out_channels,
                             int input_height, int input_width,
                             int kernel_h, int kernel_w,
                             int stride_h, int stride_w,
                             int padding_h, int padding_w,
                             int dilation_h, int dilation_w,
                             int output_height, int output_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width) return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c_out = (idx / (output_width * output_height)) % out_channels;
    int n = idx / (output_width * output_height * out_channels);

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_h = h_out * stride_h - padding_h + kh * dilation_h;
                int input_w = w_out * stride_w - padding_w + kw * dilation_w;

                if (input_h >= 0 && input_h < input_height &&
                    input_w >= 0 && input_w < input_width) {
                    int weight_offset = c_out * in_channels * kernel_h * kernel_w +
                                        c_in * kernel_h * kernel_w +
                                        kh * kernel_w + kw;
                    float weight_val = weight_data[weight_offset];

                    int input_offset = n * in_channels * input_height * input_width +
                                       c_in * input_height * input_width +
                                       input_h * input_width + input_w;
                    float input_val = input_data[input_offset];

                    sum += weight_val * input_val;
                }
            }
        }
    }
    output_data[idx] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                         int stride_h, int stride_w,
                         int padding_h, int padding_w,
                         int dilation_h, int dilation_w) {
    input = input.cuda();
    weight = weight.cuda();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int effective_kernel_h = (kernel_h - 1) * dilation_h + 1;
    int effective_kernel_w = (kernel_w - 1) * dilation_w + 1;

    int output_height = (input_height + 2 * padding_h - effective_kernel_h) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - effective_kernel_w) / stride_w + 1;

    torch::Tensor output = torch::zeros({batch_size, out_channels, output_height, output_width},
                                       torch::device("cuda"));

    int num_elements = batch_size * out_channels * output_height * output_width;

    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    conv2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_data, weight_data, output_data,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        output_height, output_width);

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                         int stride_h, int stride_w,
                         int padding_h, int padding_w,
                         int dilation_h, int dilation_w);
"""

conv2d_cuda = load_inline(
    name="conv2d_cuda",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 32, 5, 9).cuda())
        self.stride_h = 1
        self.stride_w = 1
        self.padding_h = 2
        self.padding_w = 4
        self.dilation_h = 2
        self.dilation_w = 3

    def forward(self, x):
        return conv2d_cuda.conv2d_cuda(
            x, self.weight,
            self.stride_h, self.stride_w,
            self.padding_h, self.padding_w,
            self.dilation_h, self.dilation_w
        )