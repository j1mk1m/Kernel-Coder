import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code
custom_conv3d_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_conv3d_forward(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int N, int C_in, int D, int H, int W,
    int C_out, int Kd, int Kh, int Kw,
    int D_out, int H_out, int W_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N * C_out * D_out * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c_out = (idx / (W_out * H_out * D_out)) % C_out;
    int n = idx / (C_out * D_out * H_out * W_out);

    scalar_t sum = 0.0;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < Kd; ++kd) {
            for (int kh = 0; kh < Kh; ++kh) {
                for (int kw = 0; kw < Kw; ++kw) {
                    int input_d = d_out + kd;
                    int input_h = h_out + kh;
                    int input_w = w_out + kw;

                    if (input_d < 0 || input_d >= D || 
                        input_h < 0 || input_h >= H || 
                        input_w < 0 || input_w >= W) {
                        continue;
                    }

                    int input_offset = 
                        n * C_in * D * H * W +
                        c_in * D * H * W +
                        input_d * H * W +
                        input_h * W +
                        input_w;

                    int weight_offset = 
                        c_out * C_in * Kd * Kh * Kw +
                        c_in * Kd * Kh * Kw +
                        kd * Kh * Kw +
                        kh * Kw +
                        kw;

                    sum += weight[weight_offset] * input[input_offset];
                }
            }
        }
    }

    int output_offset = 
        n * C_out * D_out * H_out * W_out +
        c_out * D_out * H_out * W_out +
        d_out * H_out * W_out +
        h_out * W_out +
        w_out;

    output[output_offset] = sum;
}

torch::Tensor custom_conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups) {

    int N = input.size(0);
    int C_in = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int C_out = weight.size(0);
    int Kd = weight.size(2);
    int Kh = weight.size(3);
    int Kw = weight.size(4);

    // Compute output dimensions
    int D_out = (D + 2*padding_d - dilation_d*(Kd-1) - 1) / stride_d + 1;
    int H_out = (H + 2*padding_h - dilation_h*(Kh-1) - 1) / stride_h + 1;
    int W_out = (W + 2*padding_w - dilation_w*(Kw-1) - 1) / stride_w + 1;

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    int total_threads = N * C_out * D_out * H_out * W_out;
    int threads_per_block = 256;
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "custom_conv3d_forward", ([&] {
        custom_conv3d_forward<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C_in, D, H, W,
            C_out, Kd, Kh, Kw,
            D_out, H_out, W_out);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

custom_conv3d_forward_cpp = """
torch::Tensor custom_conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups);
"""

custom_conv3d_forward = load_inline(
    name="custom_conv3d_forward",
    cpp_sources=[custom_conv3d_forward_cpp],
    cuda_sources=[custom_conv3d_forward_source],
    functions=["custom_conv3d_forward_cuda"],
    verbose=True
)

class CustomConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(CustomConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None

    def forward(self, x):
        stride_d, stride_h, stride_w = self.stride
        padding_d, padding_h, padding_w = self.padding
        dilation_d, dilation_h, dilation_w = self.dilation

        output = custom_conv3d_forward.custom_conv3d_forward_cuda(
            x,
            self.weight,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            dilation_d, dilation_h, dilation_w,
            self.groups
        )

        if self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1, 1, 1)

        return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = CustomConv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3d(x)