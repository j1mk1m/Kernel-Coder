import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_kernel_source = """
#include <torch/extension.h>
#include <vector>

template <int TILE_H, int TILE_W>
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in, int D_in,
    int C_out, int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    const int H_out = (H_in + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int W_out = (W_in + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int bc = blockIdx.w;

    int h_out = by * blockDim.y + ty;
    int w_out = bx * blockDim.x + tx;
    int d = bz;
    int c_out = bc * blockDim.z + tz;

    if (h_out >= H_out || w_out >= W_out || c_out >= C_out || d >= D_in) {
        return;
    }

    float sum = 0.0f;

    // Shared memory for input tile
    __shared__ float s_input[(TILE_H + kernel_h - 1)][(TILE_W + kernel_w - 1)];
    // Shared memory for weight tile
    __shared__ float s_weight[kernel_h * kernel_w];

    // Load input tile into shared memory
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            int w_in = w_out * stride_w - padding_w + kw * dilation_w;

            if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) {
                s_input[kh + ty][kw + tx] = 0.0f;
            } else {
                int input_offset = d * N * C_in * H_in * W_in +
                                   0 * C_in * H_in * W_in + // Assuming N=0 for simplicity
                                   (c_out % (C_in / groups)) * H_in * W_in +
                                   h_in * W_in + w_in;
                s_input[kh + ty][kw + tx] = input[input_offset];
            }
        }
    }

    // Load weight tile into shared memory
    for (int idx = tz; idx < kernel_h * kernel_w; idx += blockDim.z) {
        int kh = idx / kernel_w;
        int kw = idx % kernel_w;
        int weight_offset = c_out * (C_in / groups) * kernel_h * kernel_w +
                            (c_out % (C_in / groups)) * kernel_h * kernel_w +
                            kh * kernel_w + kw;
        s_weight[idx] = weight[weight_offset];
    }

    __syncthreads();

    // Compute the sum over the kernel
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int idx = kh * kernel_w + kw;
            sum += s_input[kh][kw] * s_weight[idx];
        }
    }

    // Write output
    int output_offset = d * N * C_out * H_out * W_out +
                        0 * C_out * H_out * W_out + // Assuming N=0
                        c_out * H_out * W_out +
                        h_out * W_out + w_out;
    output[output_offset] = sum;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int D_in = input.size(4);
    int C_out = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int H_out = (H_in + 2 * padding_h - dilation_h * (kernel_h - 1)) / stride_h + 1;
    int W_out = (W_in + 2 * padding_w - dilation_w * (kernel_w - 1)) / stride_w + 1;

    auto output = torch::empty({N, C_out, H_out, W_out, D_in}, input.options());

    dim3 threads(16, 16, 16);
    dim3 blocks(
        (W_out + threads.x - 1) / threads.x,
        (H_out + threads.y - 1) / threads.y,
        D_in,
        (C_out + threads.z - 1) / threads.z
    );

    conv3d_kernel<16, 16><<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in, D_in,
        C_out, kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );

    return output;
}
"""

custom_conv3d = load_inline(
    name="custom_conv3d",
    cpp_sources="",
    cuda_sources=conv3d_kernel_source,
    functions=["conv3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = (stride, stride, 1)
        self.padding = (padding, padding, 0)
        self.dilation = (dilation, dilation, 1)
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 
                                               kernel_size, kernel_size, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        output = custom_conv3d.conv3d_cuda(
            x, self.weight, 
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output