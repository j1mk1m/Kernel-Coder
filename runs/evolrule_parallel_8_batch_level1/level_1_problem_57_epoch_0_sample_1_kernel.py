import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const bool has_bias,
    const scalar_t* __restrict__ bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width) return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int oc = (idx / (output_height * output_width)) % out_channels;
    int b = idx / (out_channels * output_height * output_width);

    int out_channels_per_group = out_channels / groups;
    int g = oc / out_channels_per_group;
    int in_channels_per_group = in_channels / groups;
    int ic_start = g * in_channels_per_group;
    int ic_end = (g+1)*in_channels_per_group;

    scalar_t sum = 0.0;

    for (int ic = ic_start; ic < ic_end; ic++) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = (h_out + padding - kh + output_padding) / stride;
                int w_in = (w_out + padding - kw + output_padding) / stride;

                if (h_in >= 0 && h_in < input_height &&
                    w_in >= 0 && w_in < input_width) {

                    int local_ic = ic - ic_start;
                    int local_oc = oc % out_channels_per_group;

                    int weight_offset = local_ic * out_channels_per_group * kernel_size * kernel_size +
                                        local_oc * kernel_size * kernel_size +
                                        kh * kernel_size + kw;

                    scalar_t w_val = weight[weight_offset];

                    int input_offset = b * in_channels * input_height * input_width +
                                      ic * input_height * input_width +
                                      h_in * input_width + w_in;

                    sum += input[input_offset] * w_val;
                }
            }
        }
    }

    if (has_bias) {
        sum += bias[oc];
    }

    int output_offset = b * out_channels * output_height * output_width +
                        oc * output_height * output_width +
                        h_out * output_width + w_out;

    output[output_offset] = sum;
}

extern "C" {
    void conv_transpose2d_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int stride,
        int padding,
        int output_padding,
        int groups,
        torch::Tensor output
    ) {
        const int batch_size = input.size(0);
        const int in_channels = input.size(1);
        const int input_h = input.size(2);
        const int input_w = input.size(3);
        const int kernel_size = weight.size(2);
        const int out_channels = weight.size(1)*groups;

        const int output_h = (input_h - 1) * stride - 2 * padding + kernel_size + output_padding;
        const int output_w = (input_w - 1) * stride - 2 * padding + kernel_size + output_padding;

        if (!output.defined() || 
            output.size(0) != batch_size || 
            output.size(1) != out_channels || 
            output.size(2) != output_h || 
            output.size(3) != output_w) {
            output = torch::empty({batch_size, out_channels, output_h, output_w}, 
                                 input.options());
        }

        const int threads_per_block = 256;
        const int num_elements = batch_size * out_channels * output_h * output_w;
        const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
            conv_transpose2d_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                output_padding,
                groups,
                input_h,
                input_w,
                output_h,
                output_w,
                bias.defined(),
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr
            );
        }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
}
"""

conv_transpose2d_cuda = load_inline(
    name="conv_transpose2d_cuda",
    cpp_sources="",
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(
            (in_channels, out_channels // groups, kernel_size, kernel_size)))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        output = torch.empty(0, device=x.device)
        bias = self.bias_param if self.bias_param is not None else torch.empty(0)
        conv_transpose2d_cuda.conv_transpose2d_cuda(
            x, self.weight, bias,
            self.stride, self.padding, self.output_padding, self.groups, output
        )
        return output