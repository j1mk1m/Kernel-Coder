import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int H, int W, int D,
    int C_out, int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * H_out * W_out * D) return;

    int d = idx % D;
    int w_out = (idx / D) % W_out;
    int h_out = (idx / (D * W_out)) % H_out;
    int c_out = (idx / (D * W_out * H_out)) % C_out;
    int n = idx / (D * W_out * H_out * C_out);

    float sum = 0.0f;

    const int c_in_per_group = C_in / groups;
    const int c_out_per_group = C_out / groups;

    int group_idx = c_out / c_out_per_group;
    int c_out_in_group = c_out % c_out_per_group;

    for (int c_in_group = 0; c_in_group < c_in_per_group; ++c_in_group) {
        int c_in = group_idx * c_in_per_group + c_in_group;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                int w_in = w_out * stride_w - padding_w + kw * dilation_w;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int in_offset = 
                        n * C_in * H * W * D +
                        c_in * H * W * D +
                        h_in * W * D +
                        w_in * D +
                        d;

                    const float input_val = input[in_offset];

                    int weight_offset = 
                        (group_idx * c_out_per_group + c_out_in_group) * (c_in_per_group * kernel_h * kernel_w) +
                        c_in_group * kernel_h * kernel_w +
                        kh * kernel_w + kw;

                    const float weight_val = weight[weight_offset];

                    sum += input_val * weight_val;
                }
            }
        }
    }

    if (bias) {
        sum += bias[c_out];
    }

    int out_offset = 
        n * C_out * H_out * W_out * D +
        c_out * H_out * W_out * D +
        h_out * W_out * D +
        w_out * D +
        d;

    output[out_offset] = sum;
}

torch::Tensor conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_h, int stride_w, int padding_h, int padding_w,
    int dilation_h, int dilation_w, int groups
) {
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int D = input.size(4);

    const int C_out = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int H_out = (H + 2*padding_h - dilation_h*(kernel_h - 1) - 1) / stride_h + 1;
    const int W_out = (W + 2*padding_w - dilation_w*(kernel_w - 1) - 1) / stride_w + 1;
    const int D_out = D;

    auto output = torch::zeros({N, C_out, H_out, W_out, D}, input.options());

    const int total_threads = N * C_out * H_out * W_out * D;
    const int block_size = 256;
    const int grid_size = (total_threads + block_size - 1) / block_size;

    conv3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H, W, D,
        C_out, kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        H_out, W_out
    );

    cudaDeviceSynchronize();

    return output;
}
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size, 1
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.conv3d = load_inline(
            name="conv3d",
            cuda_sources=conv3d_source,
            functions=["conv3d_forward"],
            verbose=True
        )

    def forward(self, x):
        stride_h = self.stride
        stride_w = self.stride
        padding_h = self.padding
        padding_w = self.padding
        dilation_h = self.dilation
        dilation_w = self.dilation

        return self.conv3d.conv3d_forward(
            x, self.weight, self.bias if self.bias is not None else torch.empty(0),
            stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, self.groups
        )