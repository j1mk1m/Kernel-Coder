import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int out_depth,
    int out_height,
    int out_width,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_depth * out_height * out_width) return;

    int b = idx / (out_channels * out_depth * out_height * out_width);
    int remaining = idx % (out_channels * out_depth * out_height * out_width);
    int c_out = remaining / (out_depth * out_height * out_width);
    remaining %= (out_depth * out_height * out_width);
    int d_out = remaining / (out_height * out_width);
    remaining %= (out_height * out_width);
    int h_out = remaining / out_width;
    int w_out = remaining % out_width;

    float sum = 0.0;

    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d_in = (d_out - (kd * dilation - padding) + output_padding) / stride;
                int h_in = (h_out - (kh * dilation - padding) + output_padding) / stride;
                int w_in = (w_out - (kw * dilation - padding) + output_padding) / stride;

                if (d_in >= 0 && d_in < input_depth &&
                    h_in >= 0 && h_in < input_height &&
                    w_in >= 0 && w_in < input_width) {
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        int weight_offset = c_out * in_channels * kernel_size * kernel_size * kernel_size +
                                           c_in * kernel_size * kernel_size * kernel_size +
                                           kd * kernel_size * kernel_size +
                                           kh * kernel_size +
                                           kw;
                        float w = weight[weight_offset];

                        int input_offset = b * in_channels * input_depth * input_height * input_width +
                                          c_in * input_depth * input_height * input_width +
                                          d_in * input_height * input_width +
                                          h_in * input_width +
                                          w_in;
                        float in_val = input[input_offset];

                        sum += in_val * w;
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    int output_offset = b * out_channels * out_depth * out_height * out_width +
                       c_out * out_depth * out_height * out_width +
                       d_out * out_height * out_width +
                       h_out * out_width +
                       w_out;

    output[output_offset] = sum;
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   torch::Tensor bias,
                                   int kernel_size,
                                   int stride,
                                   int padding,
                                   int output_padding,
                                   int dilation,
                                   int groups) {
    if (input.device().is_cuda() != weight.device().is_cuda()) {
        throw std::runtime_error("Input and weight must be on the same device");
    }

    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    int out_depth = (input_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

    auto output = torch::empty({input.size(0), weight.size(0), out_depth, out_height, out_width}, input.options());

    int total_elements = output.numel();
    dim3 threads(256);
    dim3 blocks((total_elements + threads.x - 1) / threads.x, 1, 1);

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        input.size(0),
        in_channels,
        weight.size(0),
        input_depth,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        out_depth,
        out_height,
        out_width,
        groups
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

conv_transpose3d_cpp_source = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input,
                                   torch::Tensor weight,
                                   torch::Tensor bias,
                                   int kernel_size,
                                   int stride,
                                   int padding,
                                   int output_padding,
                                   int dilation,
                                   int groups);
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, 
                 output_padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias_param = None

        # Initialize parameters with the same initialization as PyTorch's ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

        # Compile the CUDA kernel
        self.conv_transpose3d = load_inline(
            name="conv_transpose3d",
            cpp_sources=conv_transpose3d_cpp_source,
            cuda_sources=conv_transpose3d_source,
            functions=["conv_transpose3d_cuda"],
            verbose=True,
            extra_cflags=["-DWITH_CUDA"],
            extra_cuda_cflags=["-arch=sm_80"]  # Assuming compute capability 8.0 or higher
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias_param if self.bias_param is not None else torch.empty(0)
        return self.conv_transpose3d.conv_transpose3d_cuda(
            x,
            self.weight,
            bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups
        )

def get_inputs():
    batch_size = 8
    in_channels = 48
    depth = 96
    height = 96
    width = 96
    x = torch.rand(batch_size, in_channels, depth, height, width).cuda()
    return [x]

def get_init_inputs():
    return [48, 24, 3]  # in_channels, out_channels, kernel_size