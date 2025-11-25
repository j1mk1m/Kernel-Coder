import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define CUDA kernels
deconv_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" __global__ void deconv_forward_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride, int padding, int output_padding,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    int C_out_per_group = C_out / groups;
    int group = c_out / C_out_per_group;
    int C_in_per_group = C_in / groups;
    int start_c_in = group * C_in_per_group;
    int end_c_in = (group + 1) * C_in_per_group;

    float sum = 0.0;

    for (int c_in = start_c_in; c_in < end_c_in; ++c_in) {
        int c_out_in_group = c_out % C_out_per_group;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = (h_out - kh + output_padding) / stride + padding;
                int w_in = (w_out - kw + output_padding) / stride + padding;

                if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in)
                    continue;

                int input_offset = n * C_in * H_in * W_in +
                                   c_in * H_in * W_in +
                                   h_in * W_in + w_in;
                float input_val = input[input_offset];

                int weight_offset = c_in * C_out_per_group * kernel_h * kernel_w +
                                    c_out_in_group * kernel_h * kernel_w +
                                    kh * kernel_w + kw;
                float weight_val = weight[weight_offset];

                sum += input_val * weight_val;
            }
        }
    }

    if (bias) {
        sum += bias[c_out];
    }

    int output_offset = n * C_out * H_out * W_out +
                        c_out * H_out * W_out +
                        h_out * W_out + w_out;
    output[output_offset] = sum;
}

torch::Tensor deconv_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                  int stride, int padding, int output_padding, int groups) {
    const int threads_per_block = 256;
    const int num_elements = input.size(0) * input.size(1) * input.size(2) * input.size(3);
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(1) * groups;
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int H_out = (H_in - 1) * stride - 2 * padding + kernel_h + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_w + output_padding;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    deconv_forward_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride, padding, output_padding,
        groups
    );

    return output;
}
"""

deconv_backward_input_source = """
extern "C" __global__ void deconv_backward_input_kernel(
    const float* grad_output, const float* weight,
    float* grad_input,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride, int padding, int output_padding,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_in * H_in * W_in) return;

    int w_in = idx % W_in;
    int h_in = (idx / W_in) % H_in;
    int c_in = (idx / (W_in * H_in)) % C_in;
    int n = idx / (W_in * H_in * C_in);

    int C_out_per_group = C_out / groups;
    int group = c_in / (C_in / groups);
    int start_c_out = group * C_out_per_group;
    int end_c_out = (group + 1) * C_out_per_group;

    float sum = 0.0;

    for (int c_out = start_c_out; c_out < end_c_out; ++c_out) {
        int c_out_in_group = c_out - start_c_out;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_out = kh * stride - padding + h_in - output_padding;
                int w_out = kw * stride - padding + w_in - output_padding;

                if (h_out < 0 || h_out >= H_out || w_out < 0 || w_out >= W_out)
                    continue;

                int grad_output_offset = n * C_out * H_out * W_out +
                                         c_out * H_out * W_out +
                                         h_out * W_out + w_out;
                float grad_output_val = grad_output[grad_output_offset];

                int weight_offset = c_in * C_out_per_group * kernel_h * kernel_w +
                                    c_out_in_group * kernel_h * kernel_w +
                                    kh * kernel_w + kw;
                float weight_val = weight[weight_offset];

                sum += grad_output_val * weight_val;
            }
        }
    }

    int grad_input_offset = n * C_in * H_in * W_in +
                            c_in * H_in * W_in +
                            h_in * W_in + w_in;
    grad_input[grad_input_offset] = sum;
}

torch::Tensor deconv_backward_input_cuda(
    torch::Tensor grad_output, torch::Tensor weight,
    int stride, int padding, int output_padding, int groups,
    int input_height, int input_width
) {
    const int threads_per_block = 256;
    const int num_elements = grad_output.size(0) * grad_output.size(1) * grad_output.size(2) * grad_output.size(3);
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    int N = grad_output.size(0);
    int C_in = weight.size(0);
    int C_out = grad_output.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int H_in = input_height;
    int W_in = input_width;

    auto grad_input = torch::zeros({N, C_in, H_in, W_in}, grad_output.options());

    deconv_backward_input_kernel<<<blocks, threads_per_block>>>(
        grad_output.data_ptr<float>(),
        weight.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, grad_output.size(2), grad_output.size(3),
        kernel_h, kernel_w,
        stride, padding, output_padding,
        groups
    );

    return grad_input;
}
"""

deconv_backward_weight_source = """
extern "C" __global__ void deconv_backward_weight_kernel(
    const float* input, const float* grad_output,
    float* grad_weight,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride, int padding, int output_padding,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_in * C_out * kernel_h * kernel_w) return;

    int kw = idx % kernel_w;
    int kh = (idx / kernel_w) % kernel_h;
    int c_out_in_group = (idx / (kernel_h * kernel_w)) % (C_out / groups);
    int c_in = idx / ((kernel_h * kernel_w) * (C_out / groups));

    int group = c_in / (C_in / groups);
    int start_c_out = group * (C_out / groups);
    int c_out = start_c_out + c_out_in_group;

    float sum = 0.0;

    for (int n = 0; n < N; ++n) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int h_in = (h_out - kh + output_padding) / stride + padding;
                int w_in = (w_out - kw + output_padding) / stride + padding;

                if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in)
                    continue;

                int input_offset = n * C_in * H_in * W_in +
                                   c_in * H_in * W_in +
                                   h_in * W_in + w_in;
                float input_val = input[input_offset];

                int grad_output_offset = n * C_out * H_out * W_out +
                                         c_out * H_out * W_out +
                                         h_out * W_out + w_out;
                float grad_output_val = grad_output[grad_output_offset];

                sum += input_val * grad_output_val;
            }
        }
    }

    int grad_weight_offset = c_in * (C_out / groups) * kernel_h * kernel_w +
                            c_out_in_group * kernel_h * kernel_w +
                            kh * kernel_w + kw;
    grad_weight[grad_weight_offset] = sum;
}

torch::Tensor deconv_backward_weight_cuda(
    torch::Tensor input, torch::Tensor grad_output,
    int stride, int padding, int output_padding, int groups
) {
    const int threads_per_block = 256;
    const int num_elements = input.size(1) * (grad_output.size(1)/groups) * input.size(2)*input.size(3);
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    int C_in = input.size(1);
    int C_out = grad_output.size(1);
    int kernel_h = (grad_output.size(2) - 1) * stride - 2 * padding + input.size(2) + output_padding;
    int kernel_w = (grad_output.size(3) - 1) * stride - 2 * padding + input.size(3) + output_padding;

    auto grad_weight = torch::zeros({C_in, C_out/groups, kernel_h, kernel_w}, input.options());

    deconv_backward_weight_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        input.size(0), C_in, input.size(2), input.size(3),
        C_out, grad_output.size(2), grad_output.size(3),
        kernel_h, kernel_w,
        stride, padding, output_padding,
        groups
    );

    return grad_weight;
}
"""

# Compile the CUDA code
deconv_forward = load_inline(
    name="deconv_forward",
    cuda_sources=deconv_forward_source,
    functions=["deconv_forward_cuda"],
    verbose=True
)

deconv_backward_input = load_inline(
    name="deconv_backward_input",
    cuda_sources=deconv_backward_input_source,
    functions=["deconv_backward_input_cuda"],
    verbose=True
)

deconv_backward_weight = load_inline(
    name="deconv_backward_weight",
    cuda_sources=deconv_backward_weight_source,
    functions=["deconv_backward_weight_cuda"],
    verbose=True
)

class DeconvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding, groups):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        return deconv_forward.deconv_forward_cuda(
            input, weight, bias if bias is not None else torch.tensor([]),
            stride, padding, output_padding, groups
        )

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        groups = ctx.groups

        grad_input = deconv_backward_input.deconv_backward_input_cuda(
            grad_output, weight,
            stride, padding, output_padding, groups,
            input.size(2), input.size(3)
        )

        grad_weight = deconv_backward_weight.deconv_backward_weight_cuda(
            input, grad_output,
            stride, padding, output_padding, groups
        )

        grad_bias = grad_output.sum((0, 2, 3)).unsqueeze(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None, None, None

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        kernel_size = (kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights and bias like PyTorch's ConvTranspose2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.bias is not None:
            return DeconvFunction.apply(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups)
        else:
            return DeconvFunction.apply(x, self.weight, None, self.stride, self.padding, self.output_padding, self.groups)