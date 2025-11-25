import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int C_in,
    int C_out,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int K_h,
    int K_w,
    int stride,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * C_out * H_out * W_out) {
        return;
    }

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float out_val = 0.0f;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_in = (h_out + padding_h - kh + output_padding_h) / stride;
                int w_in = (w_out + padding_w - kw + output_padding_w) / stride;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int w_idx = c_in * C_out * K_h * K_w + c_out * K_h * K_w + kh * K_w + kw;
                    float w_val = weight[w_idx];

                    int in_offset = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in;
                    float in_val = input[in_offset];

                    out_val += w_val * in_val;
                }
            }
        }
    }

    int output_offset = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;
    output[output_offset] = out_val;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight,
                                   int stride, int padding_h, int padding_w, int output_padding_h, int output_padding_w) {
    int batch_size = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(1);
    int K_h = weight.size(2);
    int K_w = weight.size(3);

    int H_out = (H_in - 1) * stride - 2 * padding_h + K_h + output_padding_h;
    int W_out = (W_in - 1) * stride - 2 * padding_w + K_w + output_padding_w;

    auto output = torch::empty({batch_size, C_out, H_out, W_out}, input.options());

    const int threads_per_block = 256;
    const int num_elements = batch_size * C_out * H_out * W_out;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, C_in, C_out, H_in, W_in, H_out, W_out,
        K_h, K_w, stride, padding_h, padding_w, output_padding_h, output_padding_w
    );

    return output;
}
"""

conv_transpose2d_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding_h, int padding_w, int output_padding_h, int output_padding_w);"
)

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        if isinstance(padding, int):
            self.padding_h = padding
            self.padding_w = padding
        else:
            self.padding_h, self.padding_w = padding
        if isinstance(output_padding, int):
            self.output_padding_h = output_padding
            self.output_padding_w = output_padding
        else:
            self.output_padding_h, self.output_padding_w = output_padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert groups == 1, "Groups >1 not yet supported in this implementation"

        self.weight = torch.nn.Parameter(torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1]))

    def forward(self, x):
        return conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding_h,
            self.padding_w,
            self.output_padding_h,
            self.output_padding_w
        )