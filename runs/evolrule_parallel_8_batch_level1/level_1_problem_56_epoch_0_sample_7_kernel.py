import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Initialize weights and bias
        kh, kw = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kh, kw))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return CustomConv2dFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )

# Define the CUDA kernels
conv_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups) {

    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int h = threadIdx.y;
    int w = threadIdx.x;

    if (h >= output.size(2) || w >= output.size(3)) return;

    const int out_channels_per_group = weight.size(0) / groups;
    const int in_channels_per_group = weight.size(1);

    int group = c_out / out_channels_per_group;
    int c_out_in_group = c_out % out_channels_per_group;

    int in_channel_start = group * in_channels_per_group;

    scalar_t sum = 0;
    for (int kh = 0; kh < weight.size(2); ++kh) {
        for (int kw = 0; kw < weight.size(3); ++kw) {
            int h_in = h * stride_h - padding_h + kh * dilation_h;
            int w_in = w * stride_w - padding_w + kw * dilation_w;
            if (h_in >=0 && h_in < input.size(2) && w_in >=0 && w_in < input.size(3)) {
                for (int c_in = in_channel_start; c_in < in_channel_start + in_channels_per_group; ++c_in) {
                    sum += input[n][c_in][h_in][w_in] * weight[group * out_channels_per_group + c_out_in_group][c_in - in_channel_start][kh][kw];
                }
            }
        }
    }

    if (bias.size(0) > 0) {
        sum += bias[c_out];
    }
    output[n][c_out][h][w] = sum;
}

void conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);

    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int out_height = output.size(2);
    const int out_width = output.size(3);

    dim3 threads(out_width, out_height);
    dim3 blocks(batch_size, out_channels);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv2d_forward", ([&] {
        conv2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            groups);
    }));

    cudaDeviceSynchronize();
}
"""

# Compile the CUDA code
conv_module = load_inline(
    name='conv2d',
    cpp_sources='',
    cuda_sources=conv_forward_source,
    functions=['conv2d_forward'],
    verbose=True
)

conv2d_forward = conv_module.conv2d_forward

class CustomConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        # Compute output dimensions
        n, c_in, h_in, w_in = input.size()
        kh, kw = weight.size()[2:]
        stride_h, stride_w = stride
        padding_h, padding_w = padding
        dilation_h, dilation_w = dilation

        out_channels = weight.size(0)
        out_h = (h_in + 2 * padding_h - dilation_h * (kh - 1) - 1) // stride_h + 1
        out_w = (w_in + 2 * padding_w - dilation_w * (kw - 1) - 1) // stride_w + 1

        output = torch.zeros(n, out_channels, out_h, out_w, device=input.device)

        conv2d_forward(input, weight, bias, output,
                      stride_h, stride_w,
                      padding_h, padding_w,
                      dilation_h, dilation_w,
                      groups)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        # Compute gradients using PyTorch's autograd for simplicity
        with torch.enable_grad():
            input.requires_grad = True
            weight.requires_grad = True
            if bias is not None:
                bias.requires_grad = True

            output = CustomConv2dFunction.apply(
                input, weight, bias, stride, padding, dilation, groups
            )

            grad = torch.autograd.grad(
                outputs=output,
                inputs=(input, weight, bias),
                grad_outputs=grad_output,
                retain_graph=True,
                create_graph=True
            )

            grad_input, grad_weight, grad_bias = grad

        return grad_input, grad_weight, grad_bias, None, None, None, None