import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for depthwise convolution
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int batch_size, int in_channels, int out_channels_per_in,
    int H_in, int W_in, int H_kernel, int W_kernel,
    int stride_h, int stride_w, int pad_h, int pad_w) {

    const int H_out = (H_in + 2 * pad_h - H_kernel) / stride_h + 1;
    const int W_out = (W_in + 2 * pad_w - W_kernel) / stride_w + 1;

    CUDA_KERNEL_LOOP(index, batch_size * in_channels * H_out * W_out) {
        int w_out = index % W_out;
        int h_out = (index / W_out) % H_out;
        int c_in = (index / (W_out * H_out)) % in_channels;
        int n = index / (W_out * H_out * in_channels);

        scalar_t val = 0;
        for (int kh = 0; kh < H_kernel; ++kh) {
            for (int kw = 0; kw < W_kernel; ++kw) {
                int h_in = -pad_h + h_out * stride_h + kh;
                int w_in = -pad_w + w_out * stride_w + kw;
                // Skip out of bound indices
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    for (int c_out = 0; c_out < out_channels_per_in; ++c_out) {
                        val += input[n][c_in][h_in][w_in] *
                               weight[c_in * out_channels_per_in + c_out][0][kh][kw];
                    }
                }
            }
        }
        output[n][c_in * out_channels_per_in + c_out][h_out][w_out] = val;
    }
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels_per_in = weight.size(0) / in_channels;
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int H_kernel = weight.size(2);
    const int W_kernel = weight.size(3);

    int H_out = (H_in + 2 * pad_h - H_kernel) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - W_kernel) / stride_w + 1;
    int out_channels = in_channels * out_channels_per_in;

    auto output = torch::zeros({batch_size, out_channels, H_out, W_out}, input.options());

    const int threads = 1024;
    const int num_elements = batch_size * in_channels * H_out * W_out;
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_forward", ([&] {
        depthwise_conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            batch_size, in_channels, out_channels_per_in,
            H_in, W_in, H_kernel, W_kernel,
            stride_h, stride_w, pad_h, pad_w);
    }));

    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w);
"""

# Compile the inline CUDA code
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_forward"],
    verbose=True,
    extra_cflags=["-D__CUDA_NO_HALF_OPERATORS__"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights and other parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, 1, kernel_size, kernel_size))
        self.bias_term = nn.Parameter(torch.Tensor(out_channels)) if bias else None

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias:
            nn.init.zeros_(self.bias_term)

        self.depthwise_conv2d = depthwise_conv2d

    def forward(self, x):
        # Calculate out_channels per input channel
        out_channels_per_in = self.out_channels // self.in_channels

        # Ensure that the weights are reshaped correctly
        # Assuming weight has shape (out_channels, 1, kernel_size, kernel_size)
        # For depthwise, each input channel has its own set of filters
        # So we need to reshape weight to (in_channels * out_channels_per_in, 1, kernel_size, kernel_size)
        assert self.weight.shape[0] == self.out_channels
        assert self.weight.shape[1] == 1

        output = self.depthwise_conv2d.depthwise_conv2d_forward(
            x, self.weight, self.stride, self.stride, self.padding, self.padding
        )

        if self.bias:
            output = output + self.bias_term.view(1, -1, 1, 1)

        return output

def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]