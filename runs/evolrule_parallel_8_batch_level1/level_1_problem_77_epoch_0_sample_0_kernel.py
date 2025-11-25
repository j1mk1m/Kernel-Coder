import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_depth,
    int output_height,
    int output_width,
    bool has_bias,
    const float* __restrict__ bias
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * out_channels * output_depth * output_height * output_width) return;

    int batch = tid / (out_channels * output_depth * output_height * output_width);
    int remaining = tid % (out_channels * output_depth * output_height * output_width);
    int oc = remaining / (output_depth * output_height * output_width);
    remaining %= (output_depth * output_height * output_width);
    int d_out = remaining / (output_height * output_width);
    remaining %= (output_height * output_width);
    int h_out = remaining / output_width;
    int w_out = remaining % output_width;

    float sum = 0.0f;

    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d_in = (d_out + padding - kd * dilation);
                if (d_in % stride != 0) continue;
                d_in /= stride;
                if (d_in < 0 || d_in >= input_depth) continue;

                int h_in = (h_out + padding - kh * dilation);
                if (h_in % stride != 0) continue;
                h_in /= stride;
                if (h_in < 0 || h_in >= input_height) continue;

                int w_in = (w_out + padding - kw * dilation);
                if (w_in % stride != 0) continue;
                w_in /= stride;
                if (w_in < 0 || w_in >= input_width) continue;

                for (int ic = 0; ic < in_channels; ++ic) {
                    int w_idx = ic * out_channels * kernel_size * kernel_size * kernel_size;
                    w_idx += oc * kernel_size * kernel_size * kernel_size;
                    w_idx += kd * kernel_size * kernel_size;
                    w_idx += kh * kernel_size;
                    w_idx += kw;
                    float w = weight[w_idx];

                    int in_offset = batch * in_channels * input_depth * input_height * input_width;
                    in_offset += ic * input_depth * input_height * input_width;
                    in_offset += d_in * input_height * input_width;
                    in_offset += h_in * input_width;
                    in_offset += w_in;
                    float in_val = input[in_offset];

                    sum += in_val * w;
                }
            }
        }
    }

    if (has_bias) {
        sum += bias[oc];
    }

    int out_offset = batch * out_channels * output_depth * output_height * output_width;
    out_offset += oc * output_depth * output_height * output_width;
    out_offset += d_out * output_height * output_width;
    out_offset += h_out * output_width;
    out_offset += w_out;
    output[out_offset] = sum;
}

extern "C" __host__ torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_depth,
    int output_height,
    int output_width,
    bool has_bias,
    torch::Tensor bias
) {
    const int threads_per_block = 256;
    const int num_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    conv_transpose3d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        output_depth,
        output_height,
        output_width,
        has_bias,
        has_bias ? bias.data_ptr<float>() : nullptr
    );

    return output;
}
"""

conv_transpose3d_cuda = load_inline(
    name="conv_transpose3d_cuda",
    cpp_sources="",
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias_param = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        input_depth = x.size(2)
        input_height = x.size(3)
        input_width = x.size(4)

        output_depth = (input_depth - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1

        output = torch.empty(
            x.size(0),
            self.out_channels,
            output_depth,
            output_height,
            output_width,
            device=x.device,
            dtype=x.dtype
        )

        bias_tensor = self.bias_param if self.bias_param is not None else torch.empty(0, device=x.device)
        conv_transpose3d_cuda(
            x,
            self.weight,
            output,
            x.size(0),
            self.in_channels,
            self.out_channels,
            input_depth,
            input_height,
            input_width,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            output_depth,
            output_height,
            output_width,
            self.bias_param is not None,
            bias_tensor
        )

        return output