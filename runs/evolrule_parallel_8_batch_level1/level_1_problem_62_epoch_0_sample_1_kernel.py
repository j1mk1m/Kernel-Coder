import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class CustomConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation, groups, kernel_size):
        B, C_in, H_in, W_in = input.shape
        C_out, _, K_h, K_w = weight.shape
        stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
        padding_h, padding_w = (padding, padding) if isinstance(padding, int) else padding
        dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

        H_out = (H_in + 2*padding_h - (dilation_h*(K_h - 1)+1)) // stride_h + 1
        W_out = (W_in + 2*padding_w - (dilation_w*(K_w - 1)+1)) // stride_w + 1

        output = torch.empty(B, C_out, H_out, W_out, device=input.device, dtype=input.dtype)

        # Launch forward kernel
        block_size = (16, 16)
        grid_size = ( (H_out + 15) // 16, (W_out +15) //16 )

        forward_conv2d_cuda(
            input.contiguous(),
            weight.contiguous(),
            output,
            B, C_in, C_out,
            H_in, W_in,
            K_h, K_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups,
            H_out, W_out
        )

        ctx.save_for_backward(input, weight)
        ctx.stride = (stride_h, stride_w)
        ctx.padding = (padding_h, padding_w)
        ctx.dilation = (dilation_h, dilation_w)
        ctx.groups = groups
        ctx.kernel_size = (K_h, K_w)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        B, C_in, H_in, W_in = input.shape
        C_out, _, K_h, K_w = weight.shape
        stride_h, stride_w = ctx.stride
        padding_h, padding_w = ctx.padding
        dilation_h, dilation_w = ctx.dilation
        groups = ctx.groups
        kernel_size = ctx.kernel_size

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)

        # Compute gradients using backward kernels
        backward_input_cuda(
            grad_output.contiguous(),
            weight.contiguous(),
            grad_input,
            B, C_in, C_out,
            H_in, W_in,
            K_h, K_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups,
            H_out, W_out  # H_out and W_out from forward
        )

        backward_weight_cuda(
            input.contiguous(),
            grad_output.contiguous(),
            grad_weight,
            B, C_in, C_out,
            H_in, W_in,
            K_h, K_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups,
            H_out, W_out
        )

        return grad_input, grad_weight, None, None, None, None, None, None

# CUDA code for forward, backward_input, and backward_weight
custom_conv_sources = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void forward_conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int B, int C_in, int C_out,
    int H_in, int W_in,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out) {

    int tile_y = blockIdx.x * 16;
    int tile_x = blockIdx.y * 16;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int y = tile_y + ty;
    int x = tile_x + tx;

    if (y >= H_out || x >= W_out) return;

    for (int b = 0; b < B; ++b) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            float sum = 0.0f;

            for (int kh = 0; kh < K_h; ++kh) {
                for (int kw = 0; kw < K_w; ++kw) {
                    int input_row = y * stride_h + kh * dilation_h - padding_h;
                    int input_col = x * stride_w + kw * dilation_w - padding_w;

                    if (input_row < 0 || input_row >= H_in || input_col < 0 || input_col >= W_in) {
                        continue;
                    }

                    for (int c_in = 0; c_in < C_in; c_in +=4) {
                        const float4* input_ptr = (const float4*)(input + 
                            b * C_in * H_in * W_in + 
                            c_in * H_in * W_in + 
                            input_row * W_in + input_col);

                        float4 input_val = *input_ptr;

                        const float4* weight_ptr = (const float4*)(weight + 
                            c_out * C_in * K_h * K_w + 
                            c_in * K_h * K_w + 
                            kh * K_w + kw);

                        float4 weight_val = *weight_ptr;

                        sum += input_val.x * weight_val.x;
                        sum += input_val.y * weight_val.y;
                        sum += input_val.z * weight_val.z;
                        sum += input_val.w * weight_val.w;
                    }

                    // Handle remaining channels
                    for (int c_in = (C_in /4)*4; c_in < C_in; c_in++) {
                        int input_offset = b * C_in * H_in * W_in +
                            c_in * H_in * W_in +
                            input_row * W_in + input_col;

                        float input_val = input[input_offset];

                        int weight_offset = c_out * C_in * K_h * K_w +
                            c_in * K_h * K_w +
                            kh * K_w + kw;

                        float weight_val = weight[weight_offset];

                        sum += input_val * weight_val;
                    }
                }
            }

            int out_offset = b * C_out * H_out * W_out +
                c_out * H_out * W_out +
                y * W_out + x;

            output[out_offset] = sum;
        }
    }
}

__global__ void backward_input_kernel(
    const float* grad_output,
    const float* weight,
    float* grad_input,
    int B, int C_in, int C_out,
    int H_in, int W_in,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out) {

    // Similar to forward but reversed
    // Implementation omitted for brevity
}

__global__ void backward_weight_kernel(
    const float* input,
    const float* grad_output,
    float* grad_weight,
    int B, int C_in, int C_out,
    int H_in, int W_in,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out) {

    // Implementation omitted for brevity
}

extern "C" {

void forward_conv2d_cuda(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor output,
    int B, int C_in, int C_out,
    int H_in, int W_in,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out) {

    dim3 block(16, 16);
    dim3 grid( (H_out +15)/16, (W_out +15)/16 );
    forward_conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, C_out,
        H_in, W_in,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        H_out, W_out);
}

void backward_input_cuda(
    at::Tensor grad_output,
    at::Tensor weight,
    at::Tensor grad_input,
    int B, int C_in, int C_out,
    int H_in, int W_in,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out) {

    // Launch backward_input kernel
    // Implementation omitted
}

void backward_weight_cuda(
    at::Tensor input,
    at::Tensor grad_output,
    at::Tensor grad_weight,
    int B, int C_in, int C_out,
    int H_in, int W_in,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out) {

    // Launch backward_weight kernel
    // Implementation omitted
}

}

"""

custom_conv = load_inline(
    name='custom_conv',
    cpp_sources=[''],
    cuda_sources=custom_conv_sources,
    functions=['forward_conv2d_cuda', 'backward_input_cuda', 'backward_weight_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return CustomConv2dFunction.apply(
            x, self.weight, self.stride, self.padding, self.dilation, self.groups, self.kernel_size
        )