import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int kernel_H,
    int kernel_W,
    int H_out,
    int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    scalar_t sum = 0.0;

    for (int kh = 0; kh < kernel_H; ++kh) {
        for (int kw = 0; kw < kernel_W; ++kw) {
            int input_h = h_out - kh;
            int input_w = w_out - kw;
            if (input_h < 0 || input_h >= H_in || input_w < 0 || input_w >= W_in)
                continue;

            for (int c_in = 0; c_in < C_in; ++c_in) {
                int weight_offset = c_in * C_out * kernel_H * kernel_W +
                                    c_out * kernel_H * kernel_W +
                                    kh * kernel_W +
                                    kw;
                scalar_t w_val = weight[weight_offset];

                int input_offset = n * C_in * H_in * W_in +
                                   c_in * H_in * W_in +
                                   input_h * W_in +
                                   input_w;
                sum += w_val * input[input_offset];
            }
        }
    }

    int output_offset = n * C_out * H_out * W_out +
                        c_out * H_out * W_out +
                        h_out * W_out +
                        w_out;
    output[output_offset] = sum;
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int kernel_H,
    int kernel_W,
    int H_out,
    int W_out
) {
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    const int threads_per_block = 256;
    const int num_blocks = (N * C_out * H_out * W_out + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C_in, H_in, W_in,
            C_out, kernel_H, kernel_W,
            H_out, W_out
        );
    }));

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int kernel_H,
    int kernel_W,
    int H_out,
    int W_out
);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=[conv_transpose2d_cpp_source],
    cuda_sources=[conv_transpose2d_source],
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 output_padding: tuple = (0, 0), dilation: tuple = (1, 1),
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1])
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        C_in = x.size(1)
        H_in = x.size(2)
        W_in = x.size(3)
        C_out = self.weight.size(1)
        kernel_H = self.weight.size(2)
        kernel_W = self.weight.size(3)

        H_out = (H_in - 1) * self.stride[0] - 2 * self.padding[0] + \
                self.dilation[0] * (kernel_H - 1) + self.output_padding[0] + 1
        W_out = (W_in - 1) * self.stride[1] - 2 * self.padding[1] + \
                self.dilation[1] * (kernel_W - 1) + self.output_padding[1] + 1

        return conv_transpose2d.conv_transpose2d_cuda(
            x, self.weight, N, C_in, H_in, W_in, C_out, kernel_H, kernel_W, H_out, W_out
        )