import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# CUDA kernel for forward convolution
conv2d_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv2d_forward_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height_in,
    int width_in,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int height_out,
    int width_out) {

    int output_x = blockIdx.x;
    int output_y = blockIdx.y;
    int output_c = blockIdx.z;
    int batch_idx = threadIdx.x;

    if (batch_idx >= batch_size || output_x >= width_out || output_y >= height_out || output_c >= out_channels) {
        return;
    }

    // Compute input coordinates
    int input_x = output_x * stride_w - padding_w;
    int input_y = output_y * stride_h - padding_h;

    scalar_t acc = 0.0;
    for (int ky = 0; ky < kernel_h; ++ky) {
        int y = input_y + ky * dilation_h;
        if (y < 0 || y >= height_in) continue;
        for (int kx = 0; kx < kernel_w; ++kx) {
            int x = input_x + kx * dilation_w;
            if (x < 0 || x >= width_in) continue;
            for (int c_in = 0; c_in < in_channels / groups; ++c_in) {
                int w_offset = (output_c * (in_channels / groups) + c_in) * kernel_h * kernel_w + ky * kernel_w + kx;
                int i_offset = batch_idx * in_channels * height_in * width_in +
                    (c_in + (output_c / (out_channels / groups)) * (in_channels / groups)) * height_in * width_in +
                    y * width_in + x;
                acc += input[i_offset] * weight[w_offset];
            }
        }
    }

    int output_offset = batch_idx * out_channels * height_out * width_out +
                        output_c * height_out * width_out +
                        output_y * width_out + output_x;

    output[output_offset] = acc;
}

void conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height_in,
    int width_in,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int height_out,
    int width_out) {

    dim3 threads(batch_size); // Each thread handles a batch
    dim3 blocks(width_out, height_out, out_channels);

    conv2d_forward_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        height_out,
        width_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
}
"""

conv2d_forward_cpp = """
#include <torch/extension.h>

void conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height_in,
    int width_in,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int height_out,
    int width_out);
"""

class CustomConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        batch_size, in_channels, height_in, width_in = input.shape
        out_channels, _, kernel_h, kernel_w = weight.shape

        # Compute output dimensions
        height_out = (height_in + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
        width_out = (width_in + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1

        output = torch.zeros(batch_size, out_channels, height_out, width_out, device=input.device, dtype=input.dtype)

        # Call the CUDA kernel
        conv2d_forward_cuda(
            input.contiguous(),
            weight.contiguous(),
            output,
            batch_size,
            in_channels,
            out_channels,
            height_in,
            width_in,
            kernel_h,
            kernel_w,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            groups,
            height_out,
            width_out
        )

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        # Compute gradient w.r.t input using conv_transpose2d
        grad_input = F.conv_transpose2d(
            grad_output,
            weight,
            stride=stride,
            padding=padding,
            output_padding=0,
            groups=groups,
            dilation=dilation
        )

        # Compute gradient w.r.t weight using conv2d
        grad_weight = F.conv2d(
            input,
            grad_output,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation
        )

        # Compute gradient w.r.t bias
        grad_bias = grad_output.sum((0, 2, 3)) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None, None, None

# Load the CUDA extension inline
conv2d_forward_cuda = load_inline(
    name="conv2d_forward",
    cpp_sources=conv2d_forward_cpp,
    cuda_sources=conv2d_forward_source,
    functions=["conv2d_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        # Initialize weights and bias similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups

    def forward(self, x):
        return CustomConv2dFunction.apply(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )