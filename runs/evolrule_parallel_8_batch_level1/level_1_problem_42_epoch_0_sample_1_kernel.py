import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void maxpool2d_kernel(
    const scalar_t* input,
    scalar_t* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int H_padded,
    int W_padded
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_height * output_width) {
        return;
    }

    int w_out = idx % output_width;
    int rem = idx / output_width;
    int h_out = rem % output_height;
    rem /= output_height;
    int c = rem % channels;
    int n = rem / channels;

    scalar_t max_val = -FLT_MAX;
    int h_start = h_out * stride;
    int w_start = w_out * stride;

    for (int i = 0; i < kernel_size; ++i) {
        int h_padded = h_start + i * dilation;
        if (h_padded < 0 || h_padded >= H_padded) {
            continue;
        }
        for (int j = 0; j < kernel_size; ++j) {
            int w_padded = w_start + j * dilation;
            if (w_padded < 0 || w_padded >= W_padded) {
                continue;
            }
            int original_h = h_padded - padding;
            int original_w = w_padded - padding;
            if (original_h >= 0 && original_h < input_height &&
                original_w >= 0 && original_w < input_width) {
                int input_offset = n * channels * input_height * input_width +
                                   c * input_height * input_width +
                                   original_h * input_width +
                                   original_w;
                scalar_t val = input[input_offset];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    int output_offset = n * channels * output_height * output_width +
                        c * output_height * output_width +
                        h_out * output_width +
                        w_out;
    output[output_offset] = max_val;
}

torch::Tensor maxpool2d_cuda(torch::Tensor input,
                             int kernel_size,
                             int stride,
                             int padding,
                             int dilation) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int H_padded = input_height + 2 * padding;
    const int W_padded = input_width + 2 * padding;
    const int output_height = ((H_padded - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width = ((W_padded - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::zeros({batch_size, channels, output_height, output_width},
                              torch::device(input.device()).dtype(input.scalar_type()));

    const int threads_per_block = 256;
    const int num_elements = batch_size * channels * output_height * output_width;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "maxpool2d_cuda", ([&] {
        maxpool2d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation,
            H_padded,
            W_padded
        );
    }));

    return output;
}
"""

cpp_source = "torch::Tensor maxpool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding, int dilation);"

maxpool_cuda = load_inline(
    name="maxpool_cuda",
    cpp_sources=cpp_source,
    cuda_sources=maxpool2d_source,
    functions=["maxpool2d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return maxpool_cuda(x, self.kernel_size, self.stride, self.padding, self.dilation)

batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]