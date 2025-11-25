import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class DepthwiseConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride=1, padding=0):
        batch_size, in_channels, height, width = input.shape
        kernel_size = weight.shape[-1]
        out_height = (height + 2 * padding - kernel_size) // stride + 1
        out_width = (width + 2 * padding - kernel_size) // stride + 1

        output = torch.empty(
            batch_size,
            in_channels,
            out_height,
            out_width,
            device=input.device,
            dtype=input.dtype,
        )

        # Launch CUDA kernel
        n = output.numel()
        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        depthwise_conv2d_kernel[grid_size, block_size](
            input,
            weight,
            output,
            stride,
            padding,
            in_channels,
            height,
            width,
            kernel_size,
            out_height,
            out_width,
        )

        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        batch_size, in_channels, height, width = input.shape
        kernel_size = weight.shape[-1]
        out_height, out_width = grad_output.shape[2:]

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)

        n = grad_input.numel() + grad_weight.numel()
        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        depthwise_conv2d_backward_kernel[grid_size, block_size](
            grad_output,
            input,
            weight,
            grad_input,
            grad_weight,
            stride,
            padding,
            in_channels,
            height,
            width,
            kernel_size,
            out_height,
            out_width,
        )

        return grad_input, grad_weight, None, None

# CUDA kernel code
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    const int stride,
    const int padding,
    const int channels,
    const int H,
    const int W,
    const int kernel_size,
    const int out_H,
    const int out_W
) {
    const int batch_idx = blockIdx.x;
    const int channel_idx = blockIdx.y;
    const int output_pos = threadIdx.x;

    const int H_out = out_H;
    const int W_out = out_W;

    if (channel_idx >= channels) return;
    if (output_pos >= H_out * W_out) return;

    const int h_out = output_pos / W_out;
    const int w_out = output_pos % W_out;

    const int h_in = h_out * stride - padding;
    const int w_in = w_out * stride - padding;

    scalar_t val = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int h = h_in + kh;
            const int w = w_in + kw;
            if (h >= 0 && h < H && w >= 0 && w < W) {
                val += input[batch_idx][channel_idx][h][w] * weight[channel_idx][0][kh][kw];
            }
        }
    }
    output[batch_idx][channel_idx][h_out][w_out] = val;
}

template <typename scalar_t>
__global__ void depthwise_conv2d_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_input,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_weight,
    const int stride,
    const int padding,
    const int channels,
    const int H,
    const int W,
    const int kernel_size,
    const int out_H,
    const int out_W
) {
    const int batch_idx = blockIdx.x;
    const int channel_idx = blockIdx.y;
    const int output_pos = threadIdx.x;

    const int H_out = out_H;
    const int W_out = out_W;

    if (channel_idx >= channels) return;
    if (output_pos >= H_out * W_out) return;

    const int h_out = output_pos / W_out;
    const int w_out = output_pos % W_out;

    const int h_in = h_out * stride - padding;
    const int w_in = w_out * stride - padding;

    // Compute gradient for input
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int h = h_in + kh;
            const int w = w_in + kw;
            if (h >= 0 && h < H && w >= 0 && w < W) {
                atomicAdd(
                    &grad_input[batch_idx][channel_idx][h][w],
                    grad_output[batch_idx][channel_idx][h_out][w_out] * weight[channel_idx][0][kh][kw]
                );
            }
        }
    }

    // Compute gradient for weight
    scalar_t grad_w = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int h = h_in + kh;
            const int w = w_in + kw;
            if (h >= 0 && h < H && w >= 0 && w < W) {
                grad_w += input[batch_idx][channel_idx][h][w] * grad_output[batch_idx][channel_idx][h_out][w_out];
            }
        }
    }
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            atomicAdd(
                &grad_weight[channel_idx][0][kh][kw],
                grad_w
            );
        }
    }
}

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous")

void depthwise_conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    torch::Tensor output
) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int kernel_size = weight.size(2);
    const int out_H = (H + 2 * padding - kernel_size) / stride + 1;
    const int out_W = (W + 2 * padding - kernel_size) / stride + 1;

    dim3 grid(batch_size, channels);
    dim3 block(out_H * out_W);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_forward", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<grid, block>>>(
            input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            stride,
            padding,
            channels,
            H,
            W,
            kernel_size,
            out_H,
            out_W
        );
    }));

    cudaDeviceSynchronize();
}

void depthwise_conv2d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor grad_input,
    torch::Tensor grad_weight,
    int stride,
    int padding
) {
    CHECK_CUDA(grad_output);
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(grad_output);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int kernel_size = weight.size(2);
    const int out_H = grad_output.size(2);
    const int out_W = grad_output.size(3);

    dim3 grid(batch_size, channels);
    dim3 block(out_H * out_W);

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "depthwise_conv2d_backward", ([&] {
        depthwise_conv2d_backward_kernel<scalar_t><<<grid, block>>>(
            grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            stride,
            padding,
            channels,
            H,
            W,
            kernel_size,
            out_H,
            out_W
        );
    }));

    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depthwise_conv2d_cuda_forward, "Depthwise conv2d forward (CUDA)");
    m.def("backward", &depthwise_conv2d_cuda_backward, "Depthwise conv2d backward (CUDA)");
}
"""

depthwise_conv2d_cuda = load_inline(
    name="depthwise_conv2d_cuda",
    cpp_sources="",
    cuda_sources=depthwise_conv2d_source,
    functions=["forward", "backward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.randn(
                in_channels,
                1,
                kernel_size,
                kernel_size,
                requires_grad=True,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(
                    in_channels,
                    1,
                    1,
                    requires_grad=True,
                )
            )
        else:
            self.bias = None

    def forward(self, x):
        output = depthwise_conv2d_cuda.forward(
            x, self.weight, self.stride, self.padding
        )
        if self.bias is not None:
            output += self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return output