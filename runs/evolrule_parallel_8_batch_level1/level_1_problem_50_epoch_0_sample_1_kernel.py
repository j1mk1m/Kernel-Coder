import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define T 16

__global__ void custom_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_size,
    int stride,
    int padding,
    int output_h,
    int output_w
) {
    int batch = blockIdx.x;
    int spatial_block = blockIdx.y;
    int c_out = blockIdx.z;

    int num_sbx = (output_h + T - 1) / T;
    int num_sby = (output_w + T - 1) / T;
    int sbx = spatial_block / num_sby;
    int sby = spatial_block % num_sby;

    int h_start = sbx * T;
    int w_start = sby * T;
    int h_end = min(h_start + T, output_h);
    int w_end = min(w_start + T, output_w);

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h_out = h_start + ty;
    int w_out = w_start + tx;

    if (h_out >= output_h || w_out >= output_w) return;

    float acc = 0.0;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        __shared__ float s_input[74][74];
        int input_h_start = h_start * stride - padding;
        input_h_start = max(input_h_start, 0);
        int input_w_start = w_start * stride - padding;
        input_w_start = max(input_w_start, 0);

        int input_h_end = (h_end - 1) * stride - padding + kernel_size - 1;
        input_h_end = min(input_h_end, input_h - 1);
        int input_w_end = (w_end - 1) * stride - padding + kernel_size - 1;
        input_w_end = min(input_w_end, input_w - 1);

        int rows_needed = input_h_end - input_h_start + 1;
        int cols_needed = input_w_end - input_w_start + 1;

        int t_row = threadIdx.y;
        int t_col = threadIdx.x;
        for (int row = 0; row < rows_needed; row += blockDim.y) {
            for (int col = 0; col < cols_needed; col += blockDim.x) {
                int s_row = row + t_row;
                int s_col = col + t_col;
                if (s_row < rows_needed && s_col < cols_needed) {
                    int input_row = input_h_start + s_row;
                    int input_col = input_w_start + s_col;
                    int input_offset = batch * (in_channels * input_h * input_w) +
                                      c_in * input_h * input_w +
                                      input_row * input_w + input_col;
                    s_input[s_row][s_col] = input[input_offset];
                }
            }
        }
        __syncthreads();

        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int input_row = h_out * stride - padding + kh;
                int input_col = w_out * stride - padding + kw;
                if (input_row >= input_h_start && input_row <= input_h_end &&
                    input_col >= input_w_start && input_col <= input_w_end) {
                    int sh_row = input_row - input_h_start;
                    int sh_col = input_col - input_w_start;
                    float val = s_input[sh_row][sh_col];
                    float w_val = weight[c_out * in_channels * kernel_size * kernel_size +
                                        c_in * kernel_size * kernel_size +
                                        kh * kernel_size + kw];
                    acc += val * w_val;
                }
            }
        }
        __syncthreads();
    }

    acc += bias[c_out];
    int output_offset = batch * (out_channels * output_h * output_w) +
                        c_out * output_h * output_w +
                        h_out * output_w + w_out;
    output[output_offset] = acc;
}

torch::Tensor custom_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int kernel_size
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_h = input.size(2);
    int input_w = input.size(3);
    int out_channels = weight.size(0);
    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    torch::Tensor output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    dim3 blockDim(16, 16);
    int num_sbx = (output_h + T - 1) / T;
    int num_sby = (output_w + T - 1) / T;
    int num_spatial_blocks = num_sbx * num_sby;

    dim3 gridDim(batch_size, num_spatial_blocks, out_channels);

    custom_conv2d_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_h,
        input_w,
        kernel_size,
        stride,
        padding,
        output_h,
        output_w
    );

    return output;
}
"""

cpp_source = """
torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int kernel_size);
"""

custom_conv2d = load_inline(
    name="custom_conv2d",
    cpp_sources=cpp_source,
    cuda_sources=conv2d_source,
    functions=["custom_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(96, 3, 11, 11))
        self.bias = nn.Parameter(torch.randn(96))
        self.stride = 4
        self.padding = 2
        self.kernel_size = 11

    def forward(self, x):
        return custom_conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.kernel_size
        )

def get_init_inputs():
    return [num_classes]