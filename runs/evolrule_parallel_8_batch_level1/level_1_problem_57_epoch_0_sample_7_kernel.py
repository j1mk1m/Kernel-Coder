import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    const float* bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding) {

    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int h_out = blockIdx.z;

    int tid = threadIdx.x;
    int stride_step = blockDim.x * gridDim.x;

    for (int w_out = tid; w_out < output_width; w_out += stride_step) {

        float output_val = 0.0f;

        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int numerator_h = h_out + padding - kh - output_padding;
                    int numerator_w = w_out + padding - kw - output_padding;

                    if (numerator_h % stride != 0 || numerator_w % stride != 0) {
                        continue;
                    }

                    int h_in = numerator_h / stride;
                    int w_in = numerator_w / stride;

                    if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width) {
                        continue;
                    }

                    int weight_offset = c_in * out_channels * kernel_size * kernel_size;
                    weight_offset += c_out * kernel_size * kernel_size;
                    weight_offset += kh * kernel_size + kw;
                    const float w_val = weight[weight_offset];

                    int input_offset = n * in_channels * input_height * input_width;
                    input_offset += c_in * input_height * input_width;
                    input_offset += h_in * input_width + w_in;
                    const float i_val = input[input_offset];

                    output_val += w_val * i_val;
                }
            }
        }

        if (bias != nullptr) {
            output_val += bias[c_out];
        }

        int output_offset = n * out_channels * output_height * output_width;
        output_offset += c_out * output_height * output_width;
        output_offset += h_out * output_width + w_out;
        output[output_offset] = output_val;
    }
}

torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int kernel_size) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int out_channels = weight.size(1);

    // Compute output dimensions
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, output_height, output_width}, options);

    dim3 threadsPerBlock(256, 1, 1); // Adjust threads per block as needed
    dim3 blocksPerGrid(
        batch_size, // blockIdx.x (n)
        out_channels, // blockIdx.y (c_out)
        output_height // blockIdx.z (h_out)
    );

    // Launch the kernel
    conv_transpose2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding
    );

    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int kernel_size);
"""

# Compile the CUDA kernel
conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels // groups, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights and bias using standard method
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        self.conv_transpose2d_cuda = conv_transpose2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose2d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0),
            self.stride,
            self.padding,
            self.output_padding,
            self.kernel_size
        )