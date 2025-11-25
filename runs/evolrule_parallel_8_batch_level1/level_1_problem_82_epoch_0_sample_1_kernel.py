import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class DepthWiseConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        batch_size, in_channels, in_height, in_width = input.shape
        kernel_size = weight.shape[-1]
        out_height = (in_height + 2 * padding - kernel_size) // stride + 1
        out_width = (in_width + 2 * padding - kernel_size) // stride + 1

        output = torch.empty(
            batch_size,
            in_channels,
            out_height,
            out_width,
            device=input.device,
            dtype=input.dtype,
        )

        # Call the custom CUDA kernel for forward pass
        depthwise_conv2d_forward_cuda(
            input,
            weight,
            output,
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding,
        )

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        batch_size, in_channels, in_height, in_width = input.shape
        kernel_size = weight.shape[-1]
        out_height, out_width = grad_output.shape[2:]

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = None

        if bias is not None:
            grad_bias = grad_output.sum((0, 2, 3))

        # Compute gradient w.r.t. input and weight using custom CUDA kernel
        depthwise_conv2d_backward_cuda(
            grad_output,
            input,
            weight,
            grad_input,
            grad_weight,
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding,
            out_height,
            out_width,
        )

        return grad_input, grad_weight, grad_bias, None, None

# Define the CUDA kernels using inline code
depthwise_conv2d_forward_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding) {

    const int output_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    const int channel = blockIdx.z;
    const int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.w;

    if (output_x >= output_width || output_y >= output_height) return;

    const int in_row = -padding + output_y * stride;
    const int in_col = -padding + output_x * stride;

    scalar_t val = 0.0;
    for (int ker_y = 0; ker_y < kernel_size; ++ker_y) {
        for (int ker_x = 0; ker_x < kernel_size; ++ker_x) {
            int in_r = in_row + ker_y;
            int in_c = in_col + ker_x;
            if (in_r >= 0 && in_r < in_height && in_c >= 0 && in_c < in_width) {
                int weight_idx = ker_y * kernel_size + ker_x;
                int input_offset = batch * in_channels * in_height * in_width +
                                   channel * in_height * in_width +
                                   in_r * in_width + in_c;
                int weight_offset = channel * kernel_size * kernel_size + weight_idx;
                val += input[input_offset] * weight[weight_offset];
            }
        }
    }

    int output_offset = batch * in_channels * output_height * output_width +
                        channel * output_height * output_width +
                        output_y * output_width + output_x;
    output[output_offset] = val;
}

void depthwise_conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding) {

    const int output_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        in_channels,
        batch_size
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_forward", ([&] {
        depthwise_conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding);
    }));

    cudaDeviceSynchronize();
}
"""

depthwise_conv2d_backward_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void depthwise_conv2d_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_input,
    scalar_t* __restrict__ grad_weight,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_height,
    int output_width) {

    const int channel = blockIdx.z;
    const int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch = blockIdx.w;

    if (output_x >= output_width || output_y >= output_height) return;

    // Compute gradient w.r.t input
    const int in_row = -padding + output_y * stride;
    const int in_col = -padding + output_x * stride;

    for (int ker_y = 0; ker_y < kernel_size; ++ker_y) {
        for (int ker_x = 0; ker_x < kernel_size; ++ker_x) {
            int in_r = in_row + ker_y;
            int in_c = in_col + ker_x;
            if (in_r >= 0 && in_r < in_height && in_c >= 0 && in_c < in_width) {
                int weight_idx = ker_y * kernel_size + ker_x;
                int grad_out_offset = batch * in_channels * output_height * output_width +
                                      channel * output_height * output_width +
                                      output_y * output_width + output_x;
                int input_offset = batch * in_channels * in_height * in_width +
                                   channel * in_height * in_width +
                                   in_r * in_width + in_c;
                int weight_offset = channel * kernel_size * kernel_size + weight_idx;

                atomicAdd(&grad_input[input_offset],
                    grad_output[grad_out_offset] * weight[weight_offset]);
                atomicAdd(&grad_weight[weight_offset],
                    input[input_offset] * grad_output[grad_out_offset]);
            }
        }
    }
}

void depthwise_conv2d_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor grad_input,
    torch::Tensor grad_weight,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_height,
    int output_width) {

    dim3 threads(16, 16);
    dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        in_channels,
        batch_size
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_backward", ([&] {
        depthwise_conv2d_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding,
            output_height,
            output_width);
    }));

    cudaDeviceSynchronize();
}
"""

# Compile the CUDA kernels
cuda_sources = depthwise_conv2d_forward_source + depthwise_conv2d_backward_source
cpp_source = (
    "void depthwise_conv2d_forward_cuda(torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int, int, int, int);"
    "void depthwise_conv2d_backward_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int, int, int, int, int, int);"
)

conv_kernels = load_inline(
    name="depthwise_conv_kernels",
    cpp_sources=cpp_source,
    cuda_sources=cuda_sources,
    functions=[
        "depthwise_conv2d_forward_cuda",
        "depthwise_conv2d_backward_cuda",
    ],
    verbose=True,
)

depthwise_conv2d_forward_cuda = conv_kernels.depthwise_conv2d_forward_cuda
depthwise_conv2d_backward_cuda = conv_kernels.depthwise_conv2d_backward_cuda

class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.empty(
                in_channels,
                kernel_size * kernel_size,
                dtype=torch.float32,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels, dtype=torch.float32))
        else:
            self.bias = None

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.bias is not None:
            return DepthWiseConv2dFunction.apply(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
            )
        else:
            return DepthWiseConv2dFunction.apply(
                x,
                self.weight,
                None,
                self.stride,
                self.padding,
            )