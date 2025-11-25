import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Parameters for the input and model
batch_size = 8
height = 512
width = 1024
in_channels = 64
out_channels = 128
kernel_size = 3

conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_forward_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_size, int stride, int padding, int dilation, int groups,
    int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_height * out_width) return;

    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int c_out = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_channels * out_height * out_width);

    int out_channels_per_group = out_channels / groups;
    int group = c_out / out_channels_per_group;
    int c_out_in_group = c_out % out_channels_per_group;
    int in_channels_per_group = in_channels / groups;

    float output_val = 0.0f;

    for (int c_in = 0; c_in < in_channels_per_group; ++c_in) {
        int input_c = group * in_channels_per_group + c_in;

        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_padded = h_out * stride + kh * dilation;
                int w_padded = w_out * stride + kw * dilation;

                bool valid = true;
                if (h_padded < padding || h_padded >= (input_height + padding)) valid = false;
                if (w_padded < padding || w_padded >= (input_width + padding)) valid = false;

                if (valid) {
                    int h = h_padded - padding;
                    int w = w_padded - padding;

                    int input_offset = n * in_channels * input_height * input_width +
                                      input_c * input_height * input_width +
                                      h * input_width + w;
                    float input_val = input[input_offset];

                    int global_out_c = group * out_channels_per_group + c_out_in_group;
                    int weight_offset = global_out_c * in_channels_per_group * kernel_size * kernel_size +
                                       c_in * kernel_size * kernel_size +
                                       kh * kernel_size + kw;
                    float weight_val = weight[weight_offset];

                    output_val += input_val * weight_val;
                }
            }
        }
    }

    if (bias) {
        output_val += bias[c_out];
    }

    int output_offset = n * out_channels * out_height * out_width +
                       c_out * out_height * out_width +
                       h_out * out_width + w_out;
    output[output_offset] = output_val;
}

torch::Tensor conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());

    int total_threads = batch_size * out_channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    conv2d_forward_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_size, stride, padding, dilation, groups,
        out_height, out_width
    );

    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("CUDA error in conv2d_forward");
    }

    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
);
"""

# Compile the CUDA code
conv2d = load_inline(
    name="conv2d",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv2d.weight
        bias = self.conv2d.bias if self.conv2d.bias is not None else torch.empty(0)
        stride = self.conv2d.stride[0]
        padding = self.conv2d.padding[0]
        dilation = self.conv2d.dilation[0]
        groups = self.conv2d.groups

        return conv2d.conv2d_forward_cuda(x, weight, bias, stride, padding, dilation, groups)

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]