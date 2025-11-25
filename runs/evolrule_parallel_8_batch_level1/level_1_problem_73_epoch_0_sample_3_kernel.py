import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

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
        weight_shape = (in_channels, out_channels // groups, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if self.bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        # Initialize weights like PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)

        # Define and compile the CUDA kernel
        conv_transpose3d_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void conv_transpose3d_kernel(
            const scalar_t* __restrict__ input,
            const scalar_t* __restrict__ weight,
            const scalar_t* __restrict__ bias,
            scalar_t* __restrict__ output,
            int batch_size,
            int in_channels,
            int out_channels,
            int kernel_size,
            int stride,
            int padding,
            int output_padding,
            int groups,
            int input_depth,
            int input_height,
            int input_width,
            int output_depth,
            int output_height,
            int output_width
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= batch_size * out_channels * output_depth * output_height * output_width)
                return;

            int w_out = idx % output_width;
            int h_out = (idx / output_width) % output_height;
            int d_out = (idx / (output_width * output_height)) % output_depth;
            int out_c = (idx / (output_width * output_height * output_depth)) % out_channels;
            int b = idx / (out_channels * output_depth * output_height * output_width);

            int out_per_group = out_channels / groups;
            int g = out_c / out_per_group;
            int in_per_group = in_channels / groups;
            int in_c_start = g * in_per_group;
            int out_c_group = out_c % out_per_group;

            scalar_t acc = 0.0;

            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        // Compute input indices considering output_padding
                        int d_in = (d_out + padding - kd - output_padding) / stride;
                        int h_in = (h_out + padding - kh - output_padding) / stride;
                        int w_in = (w_out + padding - kw - output_padding) / stride;

                        if (d_in < 0 || d_in >= input_depth) continue;
                        if (h_in < 0 || h_in >= input_height) continue;
                        if (w_in < 0 || w_in >= input_width) continue;

                        // Iterate over input channels in the group
                        for (int in_c = in_c_start; in_c < in_c_start + in_per_group; ++in_c) {
                            // Compute weight index
                            int weight_offset = in_c * (out_per_group * kernel_size * kernel_size * kernel_size) 
                                + out_c_group * (kernel_size * kernel_size * kernel_size) 
                                + kd * kernel_size * kernel_size 
                                + kh * kernel_size 
                                + kw;

                            // Get input value
                            int input_offset = b * in_channels * input_depth * input_height * input_width
                                + in_c * input_depth * input_height * input_width
                                + d_in * input_height * input_width
                                + h_in * input_width
                                + w_in;

                            acc += input[input_offset] * weight[weight_offset];
                        }
                    }
                }
            }

            // Add bias
            if (bias != nullptr) {
                acc += bias[out_c];
            }

            // Write to output
            output[idx] = acc;
        }

        template <typename scalar_t>
        torch::Tensor conv_transpose3d_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int batch_size,
            int in_channels,
            int out_channels,
            int kernel_size,
            int stride,
            int padding,
            int output_padding,
            int groups,
            int input_depth,
            int input_height,
            int input_width,
            int output_depth,
            int output_height,
            int output_width
        ) {
            const int threads = 256;
            const int elements = batch_size * out_channels * output_depth * output_height * output_width;
            const int blocks = (elements + threads - 1) / threads;

            auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, input.options());

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_cuda", ([&] {
                conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    output_padding,
                    groups,
                    input_depth,
                    input_height,
                    input_width,
                    output_depth,
                    output_height,
                    output_width
                );
            }));

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error: %s\\n", cudaGetErrorString(err));
            }
            return output;
        }
        """

        conv_transpose3d_cpp_source = """
        #include <torch/extension.h>

        torch::Tensor conv_transpose3d_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int batch_size,
            int in_channels,
            int out_channels,
            int kernel_size,
            int stride,
            int padding,
            int output_padding,
            int groups,
            int input_depth,
            int input_height,
            int input_width,
            int output_depth,
            int output_height,
            int output_width
        );
        """

        # Compile the CUDA kernel
        self.conv_transpose3d = load_inline(
            name="conv_transpose3d",
            cpp_sources=conv_transpose3d_cpp_source,
            cuda_sources=conv_transpose3d_source,
            functions=["conv_transpose3d_cuda"],
            verbose=True
        )

    def forward(self, x):
        batch_size, _, input_depth, input_height, input_width = x.size()
        output_depth = (input_depth - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_height = (input_height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        output_width = (input_width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Call the CUDA kernel
        output = self.conv_transpose3d.conv_transpose3d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias_param if self.bias else torch.tensor([]),
            batch_size,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width
        )

        return output