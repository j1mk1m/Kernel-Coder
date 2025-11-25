import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv3d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int output_depth,
    const int output_height,
    const int output_width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * output_depth * output_height * output_width)
        return;

    int w_out = index % output_width;
    int h_out = (index / output_width) % output_height;
    int d_out = (index / (output_width * output_height)) % output_depth;
    int c_out = (index / (output_width * output_height * output_depth)) % out_channels;
    int n = index / (out_channels * output_depth * output_height * output_width);

    int group = c_out / (out_channels / groups);
    int out_channels_per_group = out_channels / groups;
    int c_out_in_group = c_out % out_channels_per_group;

    int in_channels_per_group = in_channels / groups;
    int c_in_start = group * in_channels_per_group;

    scalar_t sum = 0;

    for (int c_in = 0; c_in < in_channels_per_group; ++c_in) {
        int in_channel = c_in_start + c_in;
        for (int kd = 0; kd < kernel_size; ++kd) {
            int input_d = d_out * stride - padding + kd * dilation;
            if (input_d < 0 || input_d >= input_depth) continue;
            for (int kh = 0; kh < kernel_size; ++kh) {
                int input_h = h_out * stride - padding + kh * dilation;
                if (input_h < 0 || input_h >= input_height) continue;
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int input_w = w_out * stride - padding + kw * dilation;
                    if (input_w < 0 || input_w >= input_width) continue;

                    int weight_offset = (c_out_in_group * in_channels_per_group + c_in) * 
                                        (kernel_size * kernel_size * kernel_size) + 
                                        kd * kernel_size * kernel_size + kh * kernel_size + kw;

                    int input_offset = n * in_channels * input_depth * input_height * input_width +
                                       in_channel * input_depth * input_height * input_width +
                                       input_d * input_height * input_width +
                                       input_h * input_width + input_w;

                    sum += weight[weight_offset] * input[input_offset];
                }
            }
        }
    }

    int output_offset = n * out_channels * output_depth * output_height * output_width +
                        c_out * output_depth * output_height * output_width +
                        d_out * output_height * output_width +
                        h_out * output_width + w_out;

    output[output_offset] = sum;
}

template <typename scalar_t>
__global__ void conv3d_backward_weight_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_weight,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int output_depth,
    const int output_height,
    const int output_width) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= out_channels * in_channels * kernel_size * kernel_size * kernel_size)
        return;

    int kw = index % kernel_size;
    index /= kernel_size;
    int kh = index % kernel_size;
    index /= kernel_size;
    int kd = index % kernel_size;
    index /= kernel_size;
    int c_in_group = index % (in_channels / groups);
    int c_out_group = index / (in_channels / groups);

    int group = c_out_group / (out_channels / groups);
    int c_out = group * (out_channels / groups) + c_out_group;

    scalar_t grad = 0;

    for (int n = 0; n < batch_size; ++n) {
        for (int d_out = 0; d_out < output_depth; ++d_out) {
            for (int h_out = 0; h_out < output_height; ++h_out) {
                for (int w_out = 0; w_out < output_width; ++w_out) {
                    int d_in = d_out * stride - padding + kd * dilation;
                    if (d_in < 0 || d_in >= input_depth) continue;
                    int h_in = h_out * stride - padding + kh * dilation;
                    if (h_in < 0 || h_in >= input_height) continue;
                    int w_in = w_out * stride - padding + kw * dilation;
                    if (w_in < 0 || w_in >= input_width) continue;

                    int input_offset = n * in_channels * input_depth * input_height * input_width +
                                       (group * (in_channels / groups) + c_in_group) * input_depth * input_height * input_width +
                                       d_in * input_height * input_width +
                                       h_in * input_width + w_in;

                    int grad_output_offset = n * out_channels * output_depth * output_height * output_width +
                                             c_out * output_depth * output_height * output_width +
                                             d_out * output_height * output_width +
                                             h_out * output_width + w_out;

                    grad += input[input_offset] * grad_output[grad_output_offset];
                }
            }
        }
    }

    int weight_index = c_out * in_channels * kernel_size * kernel_size * kernel_size +
                       (group * (in_channels / groups) + c_in_group) * kernel_size * kernel_size * kernel_size +
                       kd * kernel_size * kernel_size + kh * kernel_size + kw;

    grad_weight[weight_index] = grad;
}

template <typename scalar_t>
__global__ void conv3d_backward_input_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_input,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int output_depth,
    const int output_height,
    const int output_width) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * in_channels * input_depth * input_height * input_width)
        return;

    int w_in = index % input_width;
    int h_in = (index / input_width) % input_height;
    int d_in = (index / (input_width * input_height)) % input_depth;
    int c_in = (index / (input_width * input_height * input_depth)) % in_channels;
    int n = index / (in_channels * input_depth * input_height * input_width);

    int group = c_in / (in_channels / groups);
    int in_channels_per_group = in_channels / groups;
    int c_in_group = c_in % in_channels_per_group;

    scalar_t grad = 0;

    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d_out = (d_in - kd * dilation + padding) / stride;
                if (d_out < 0 || d_out >= output_depth) continue;
                int h_out = (h_in - kh * dilation + padding) / stride;
                if (h_out < 0 || h_out >= output_height) continue;
                int w_out = (w_in - kw * dilation + padding) / stride;
                if (w_out < 0 || w_out >= output_width) continue;

                for (int c_out_group = 0; c_out_group < (out_channels / groups); ++c_out_group) {
                    int c_out = group * (out_channels / groups) + c_out_group;
                    int weight_offset = (c_out_group * in_channels_per_group + c_in_group) * kernel_size * kernel_size * kernel_size +
                                        kd * kernel_size * kernel_size + kh * kernel_size + kw;

                    int grad_output_offset = n * out_channels * output_depth * output_height * output_width +
                                             c_out * output_depth * output_height * output_width +
                                             d_out * output_height * output_width +
                                             h_out * output_width + w_out;

                    grad += weight[weight_offset] * grad_output[grad_output_offset];
                }
            }
        }
    }

    int grad_input_offset = n * in_channels * input_depth * input_height * input_width +
                           c_in * input_depth * input_height * input_width +
                           d_in * input_height * input_width +
                           h_in * input_width + w_in;

    grad_input[grad_input_offset] = grad;
}

torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation, int groups) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);

    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int output_depth = (input_depth + 2 * padding - dilation * (kernel_size - 1)) / stride + 1;
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1)) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1)) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

    const int threads = 256;
    const int num_elements = output.numel();
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_forward", ([&] {
        conv3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_depth, input_height, input_width,
            kernel_size, stride, padding, dilation, groups,
            output_depth, output_height, output_width
        );
    }));

    if (bias.defined()) {
        output = output + bias.view({1, -1, 1, 1, 1});
    }

    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conv3d_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int groups,
    torch::Tensor bias) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);

    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int output_depth = grad_output.size(2);
    const int output_height = grad_output.size(3);
    const int output_width = grad_output.size(4);

    auto grad_input = torch::zeros_like(input);
    const int grad_input_size = grad_input.numel();
    const int grad_input_blocks = (grad_input_size + 256 - 1) / 256;

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv3d_backward_input", ([&] {
        conv3d_backward_input_kernel<scalar_t><<<grad_input_blocks, 256>>>(
            grad_output.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_depth, input_height, input_width,
            kernel_size, stride, padding, dilation, groups,
            output_depth, output_height, output_width
        );
    }));

    auto grad_weight = torch::zeros_like(weight);
    const int grad_weight_size = grad_weight.numel();
    const int grad_weight_blocks = (grad_weight_size + 256 - 1) / 256;

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "conv3d_backward_weight", ([&] {
        conv3d_backward_weight_kernel<scalar_t><<<grad_weight_blocks, 256>>>(
            input.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_depth, input_height, input_width,
            kernel_size, stride, padding, dilation, groups,
            output_depth, output_height, output_width
        );
    }));

    torch::Tensor grad_bias;
    if (bias.defined()) {
        grad_bias = grad_output.sum({0, 2, 3, 4});
    } else {
        grad_bias = torch::Tensor();
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv3d_forward, "3D convolution forward");
    m.def("backward", &conv3d_backward, "3D convolution backward");
}
"""

conv3d_cpp_source = (
    "torch::Tensor forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation, int groups);"
    "std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation, int groups, torch::Tensor bias);"
)

conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["forward", "backward"],
    verbose=True,
)

class MyConv3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        return conv3d.forward(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = conv3d.backward(
            grad_output.contiguous(),
            input.contiguous(),
            weight.contiguous(),
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
            bias if bias is not None else torch.empty(0),
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return MyConv3dFunction.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)