import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Transposed Convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_3d_kernel(float* input, float* weight, float* output, int N, int C_in, int D_in, int H_in, int W_in, int C_out, int D_out, int H_out, int W_out, int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w) {
    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int d_out = blockIdx.z;
    int h_out = blockIdx.w;
    int w_out = blockIdx.x % W_out;

    int d_in = (d_out - pad_d) / stride_d;
    int h_in = (h_out - pad_h) / stride_h;
    int w_in = (w_out - pad_w) / stride_w;

    if (n >= N || c_out >= C_out || d_out >= D_out || h_out >= H_out || w_out >= W_out || d_in < 0 || d_in >= D_in || h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) {
        return;
    }

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int dd = 0; dd < stride_d; ++dd) {
            for (int hh = 0; hh < stride_h; ++hh) {
                for (int ww = 0; ww < stride_w; ++ww) {
                    int di = d_in + dd;
                    int hi = h_in + hh;
                    int wi = w_in + ww;
                    int idx_input = ((n * C_in + c_in) * D_in + di) * H_in + hi * W_in + wi;
                    int idx_weight = ((c_out * C_in + c_in) * stride_d + dd) * stride_h + hh * stride_w + ww;
                    sum += input[idx_input] * weight[idx_weight];
                }
            }
        }
    }
    int idx_output = ((n * C_out + c_out) * D_out + d_out) * H_out + h_out * W_out + w_out;
    output[idx_output] = sum;
}

void conv_transpose_3d_forward_cuda(float* input, float* weight, float* output, int N, int C_in, int D_in, int H_in, int W_in, int C_out, int D_out, int H_out, int W_out, int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w) {
    dim3 grid(D_out, H_out, W_out);
    dim3 block(C_out, N);
    conv_transpose_3d_kernel<<<grid, block>>>(input, weight, output, N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w);
}
"""

conv_transpose_3d_cpp_source = (
    "void conv_transpose_3d_forward_cuda(float* input, float* weight, float* output, int N, int C_in, int D_in, int H_in, int W_in, int C_out, int D_out, int H_out, int W_out, int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w);"
)

# Compile the inline CUDA code for 3D Transposed Convolution
conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose_3d
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        N, C_in, D_in, H_in, W_in = x.size()
        C_out, _, _, _, _ = self.conv_transpose.weight.size()

        output = torch.zeros(N, C_out, D_in * stride[0], H_in * stride[1], W_in * stride[2]).cuda()
        self.conv_transpose_forward_cuda(x.contiguous().data_ptr(), self.conv_transpose.weight.contiguous().data_ptr(), output.contiguous().data_ptr(), N, C_in, D_in, H_in, W_in, C_out, D_in * stride[0], H_in * stride[1], W_in * stride[2], stride[0], stride[1], stride[2], padding[0], padding[1], padding[2])
        x = output

        x = self.batch_norm(x)
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        return x


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]