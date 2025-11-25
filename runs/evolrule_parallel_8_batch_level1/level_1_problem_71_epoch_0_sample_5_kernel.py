import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels, int kernel_size,
    int input_height, int input_width,
    int output_height, int output_width,
    int stride, int padding, int output_padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_height * output_width)
        return;

    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int c_out = (idx / (output_height * output_width)) % out_channels;
    int n = idx / (out_channels * output_height * output_width);

    float acc = 0.0;
    if (bias != nullptr) {
        acc = bias[c_out];
    }

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = (h_out - kh + padding) / stride - output_padding;
                int w_in = (w_out - kw + padding) / stride - output_padding;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int weight_offset = c_in * out_channels * kernel_size * kernel_size +
                                        c_out * kernel_size * kernel_size +
                                        kh * kernel_size + kw;
                    float w_val = weight[weight_offset];

                    int input_offset = n * in_channels * input_height * input_width +
                                       c_in * input_height * input_width +
                                       h_in * input_width + w_in;
                    float in_val = input[input_offset];

                    acc += in_val * w_val;
                }
            }
        }
    }

    int output_offset = n * out_channels * output_height * output_width +
                        c_out * output_height * output_width +
                        h_out * output_width + w_out;
    output[output_offset] = acc;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int stride, int padding, int output_padding) {
    auto device = input.device();
    weight = weight.to(device);
    if (bias.defined()) {
        bias = bias.to(device);
    }

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    int total_threads = batch_size * out_channels * output_height * output_width;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, kernel_size,
        input_height, input_width,
        output_height, output_width,
        stride, padding, output_padding
    );

    return output;
}
"""

cpp_source = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                   int stride, int padding, int output_padding);
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups  # Currently unhandled in kernel
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

        self.conv_transpose2d = load_inline(
            name="conv_transpose2d",
            cuda_sources=cuda_source,
            cpp_sources=cpp_source,
            functions=["conv_transpose2d_cuda"],
            verbose=True
        )

    def forward(self, x):
        bias = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return self.conv_transpose2d.conv_transpose2d_cuda(
            x, self.weight, bias,
            self.stride, self.padding, self.output_padding
        )