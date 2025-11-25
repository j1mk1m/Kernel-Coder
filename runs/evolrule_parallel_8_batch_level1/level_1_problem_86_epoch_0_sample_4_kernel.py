import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define CUDA kernels for depthwise convolution
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv_forward(
    const float* input,
    const float* kernel,
    float* output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_height,
    int output_width) {

    int h_out = blockIdx.x;
    int w_out = blockIdx.y;
    int batch = threadIdx.x;
    int channel = threadIdx.y;

    if (batch >= batch_size || channel >= in_channels) return;

    float sum = 0.0f;
    for (int k_h = 0; k_h < kernel_size; ++k_h) {
        for (int k_w = 0; k_w < kernel_size; ++k_w) {
            int h_in = h_out * stride - padding + k_h * dilation;
            int w_in = w_out * stride - padding + k_w * dilation;
            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                int input_offset = batch * in_channels * input_height * input_width
                                 + channel * input_height * input_width
                                 + h_in * input_width + w_in;
                sum += input[input_offset] * kernel[channel * kernel_size * kernel_size + k_h * kernel_size + k_w];
            }
        }
    }

    int output_offset = batch * in_channels * output_height * output_width
                      + channel * output_height * output_width
                      + h_out * output_width + w_out;
    output[output_offset] = sum;
}

torch::Tensor depthwise_conv_forward_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    int stride,
    int padding,
    int dilation) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int kernel_size = kernel.size(2);

    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::zeros({batch_size, in_channels, output_height, output_width}, input.options());

    dim3 blockDim(batch_size, in_channels);
    dim3 grid(output_height, output_width);

    depthwise_conv_forward<<<grid, blockDim>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, input_height, input_width,
        kernel_size, stride, padding, dilation,
        output_height, output_width);

    return output;
}
"""

depthwise_conv_cpp_source = """
torch::Tensor depthwise_conv_forward_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    int stride,
    int padding,
    int dilation);
"""

# Compile depthwise convolution kernels
depthwise_conv = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv_forward_cuda"],
    verbose=True,
)

# Custom autograd function for depthwise convolution
class DepthwiseConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel, stride, padding, dilation):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.save_for_backward(input, kernel)
        return depthwise_conv.depthwise_conv_forward_cuda(input, kernel, stride, padding, dilation)

    @staticmethod
    def backward(ctx, grad_output):
        input, kernel = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation

        # Compute gradients using adjoint kernels (simplified placeholder)
        grad_input = torch.zeros_like(input)
        grad_kernel = torch.zeros_like(kernel)

        # Gradient calculation requires additional kernels, which would be implemented similarly
        return grad_input, grad_kernel, None, None, None

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise_weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        depthwise_out = DepthwiseConvFunction.apply(
            x, self.depthwise_weight, self.stride, self.padding, self.dilation
        )
        return self.pointwise(depthwise_out)