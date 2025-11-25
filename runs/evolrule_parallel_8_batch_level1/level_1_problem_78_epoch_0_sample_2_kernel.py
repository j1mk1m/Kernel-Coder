import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_forward(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int H_in,
    int W_in,
    int H_out,
    int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * H_out * W_out)
        return;

    int x = idx % W_out;
    int y = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % out_channels;
    int n = idx / (out_channels * H_out * W_out);

    float acc = 0.0;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int y_in = (y - kh + padding_h) / stride_h;
                int x_in = (x - kw + padding_w) / stride_w;

                if (y_in >= 0 && y_in < H_in && x_in >= 0 && x_in < W_in) {
                    int weight_offset = c_in * out_channels * kernel_h * kernel_w +
                                        c_out * kernel_h * kernel_w +
                                        kh * kernel_w + kw;
                    float w = weight[weight_offset];

                    int input_offset = n * in_channels * H_in * W_in +
                                       c_in * H_in * W_in +
                                       y_in * W_in + x_in;
                    float in_val = input[input_offset];
                    acc += w * in_val;
                }
            }
        }
    }
    int output_offset = n * out_channels * H_out * W_out +
                       c_out * H_out * W_out +
                       y * W_out + x;
    output[output_offset] = acc;
}

torch::Tensor conv_transpose_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int kernel_h,
    int kernel_w,
    int H_in,
    int W_in,
    int H_out,
    int W_out
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(1);

    auto output = torch::zeros({batch_size, out_channels, H_out, W_out}, input.options());

    int num_threads = batch_size * out_channels * H_out * W_out;
    int block_size = 256;
    int num_blocks = (num_threads + block_size - 1) / block_size;

    conv_transpose_forward<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        H_in,
        W_in,
        H_out,
        W_out
    );

    return output;
}
"""

conv_transpose_cpp_source = """
torch::Tensor conv_transpose_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int kernel_h,
    int kernel_w,
    int H_in,
    int W_in,
    int H_out,
    int W_out
);
"""

conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride_h, self.stride_w = stride
        self.padding_h, self.padding_w = padding
        self.kernel_h, self.kernel_w = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize weights with correct dimensions (in_channels, out_channels, kh, kw)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, self.kernel_h, self.kernel_w))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        H_in, W_in = x.size(2), x.size(3)
        # Compute output dimensions
        H_out = (H_in - 1) * self.stride_h - 2 * self.padding_h + self.kernel_h
        W_out = (W_in - 1) * self.stride_w - 2 * self.padding_w + self.kernel_w

        # Call the CUDA kernel
        output = conv_transpose.conv_transpose_forward_cuda(
            x,
            self.weight,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.kernel_h,
            self.kernel_w,
            H_in,
            W_in,
            H_out,
            W_out
        )
        return output