import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void im2col_kernel(const float* input_padded, float* col,
                             int N, int C, int H_padded, int W_padded,
                             int stride_h, int stride_w,
                             int dilation_h, int dilation_w,
                             int Kh, int Kw,
                             int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * Kh * Kw * H_out * W_out) return;

    int sample = idx / (C * Kh * Kw * H_out * W_out);
    int remainder = idx % (C * Kh * Kw * H_out * W_out);

    int row = remainder / (H_out * W_out);
    int col_idx = remainder % (H_out * W_out);

    int output_h = col_idx / W_out;
    int output_w = col_idx % W_out;

    int c = row / (Kh * Kw);
    int kh = (row % (Kh * Kw)) / Kw;
    int kw = row % Kw;

    int input_h_start = output_h * stride_h;
    int input_w_start = output_w * stride_w;

    int input_h = input_h_start + kh * dilation_h;
    int input_w = input_w_start + kw * dilation_w;

    int input_offset = sample * C * H_padded * W_padded + c * H_padded * W_padded;
    input_offset += input_h * W_padded + input_w;

    float val = input_padded[input_offset];
    int col_offset = sample * C * Kh * Kw * H_out * W_out + row * H_out * W_out + col_idx;
    col[col_offset] = val;
}

torch::Tensor custom_conv2d(torch::Tensor input_padded, torch::Tensor weight,
                           int stride_h, int stride_w,
                           int dilation_h, int dilation_w) {
    input_padded = input_padded.contiguous();
    weight = weight.contiguous();

    int N = input_padded.size(0);
    int C = input_padded.size(1);
    int H_padded = input_padded.size(2);
    int W_padded = input_padded.size(3);

    int F = weight.size(0);
    int Kh = weight.size(2);
    int Kw = weight.size(3);

    int H_out = (H_padded - dilation_h*(Kh-1) - 1) / stride_h + 1;
    int W_out = (W_padded - dilation_w*(Kw-1) - 1) / stride_w + 1;

    int K = C * Kh * Kw;
    int cols = H_out * W_out;

    auto col = torch::empty({N, K, cols}, input_padded.options());

    int total_elements = N * K * cols;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    im2col_kernel<<<blocks, threads>>>(input_padded.data_ptr<float>(),
                                      col.data_ptr<float>(),
                                      N, C, H_padded, W_padded,
                                      stride_h, stride_w,
                                      dilation_h, dilation_w,
                                      Kh, Kw,
                                      H_out, W_out);

    auto weight_reshaped = weight.view({F, K});
    auto output = torch::matmul(weight_reshaped, col.transpose(1, 2));
    output = output.transpose(1, 2).view({N, F, H_out, W_out});

    return output;
}
"""

conv2d_cuda = load_inline(
    name="conv2d_cuda",
    cuda_sources=conv2d_source,
    functions=["custom_conv2d"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups  # Currently groups=1 is assumed
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        padding_h = self.padding
        padding_w = self.padding
        input_padded = F.pad(x, (padding_w, padding_w, padding_h, padding_h))

        stride_h = self.stride
        stride_w = self.stride
        dilation_h = self.dilation
        dilation_w = self.dilation

        output = conv2d_cuda.custom_conv2d(
            input_padded, self.weight,
            stride_h, stride_w,
            dilation_h, dilation_w
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output