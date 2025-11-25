import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(
            in_channels // groups,
            out_channels // groups,
            kernel_size,
            kernel_size
        ))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # CUDA kernel definition
        cuda_src = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void conv_transpose2d_kernel(
            const float* input,
            const float* weight,
            const float* bias,
            float* output,
            int batch_size,
            int in_channels,
            int out_channels,
            int kernel_size,
            int stride,
            int padding,
            int output_padding,
            int input_height,
            int input_width,
            int output_height,
            int output_width,
            int groups,
            int has_bias
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx >= batch_size * out_channels * output_height * output_width) return;

            // Compute indices
            int w_out = idx % output_width;
            int h_out_temp = idx / output_width;
            int h_out = h_out_temp % output_height;
            int remaining = h_out_temp / output_height;
            int c_out = remaining % out_channels;
            int n = remaining / out_channels;

            // Calculate group
            int out_per_group = out_channels / groups;
            int group = c_out / out_per_group;
            int c_out_in_group = c_out % out_per_group;

            int in_per_group = in_channels / groups;
            int start_c_in = group * in_per_group;
            int end_c_in = (group + 1) * in_per_group;

            float sum = 0.0f;

            // Iterate over input channels in the group
            for (int c_in = start_c_in; c_in < end_c_in; ++c_in) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int h_in = (h_out - kh + padding + output_padding) / stride;
                        int w_in = (w_out - kw + padding + output_padding) / stride;

                        if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width) {
                            continue;
                        }

                        // Calculate weight index within the group
                        int c_in_in_group = c_in - start_c_in;
                        int weight_offset = c_in_in_group * out_per_group * kernel_size * kernel_size +
                                            c_out_in_group * kernel_size * kernel_size +
                                            kh * kernel_size +
                                            kw;

                        float w_val = weight[weight_offset];

                        // Input offset
                        int input_offset = n * in_channels * input_height * input_width +
                                           c_in * input_height * input_width +
                                           h_in * input_width + w_in;

                        float in_val = input[input_offset];

                        sum += w_val * in_val;
                    }
                }
            }

            // Add bias
            if (has_bias) {
                sum += bias[c_out];
            }

            // Write to output
            int output_offset = n * out_channels * output_height * output_width +
                               c_out * output_height * output_width +
                               h_out * output_width + w_out;

            output[output_offset] = sum;
        }

        torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                           torch::Tensor weight,
                                           torch::Tensor bias,
                                           int stride,
                                           int padding,
                                           int output_padding,
                                           int kernel_size,
                                           int groups) {
            const auto batch_size = input.size(0);
            const auto in_channels = input.size(1);
            const auto input_height = input.size(2);
            const auto input_width = input.size(3);

            const auto out_channels = weight.size(0) * groups;

            // Compute output dimensions
            const auto output_height = (input_height - 1) * stride + kernel_size - 2 * padding + output_padding;
            const auto output_width = (input_width - 1) * stride + kernel_size - 2 * padding + output_padding;

            auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

            const int threads_per_block = 256;
            const int num_elements = output.numel();
            const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

            // Launch kernel
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
                conv_transpose2d_kernel<<<blocks_per_grid, threads_per_block>>>(
                    input.data_ptr<float>(),
                    weight.data_ptr<float>(),
                    (bias.defined() ? bias.data_ptr<float>() : nullptr),
                    output.data_ptr<float>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    output_padding,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    groups,
                    bias.defined() ? 1 : 0
                );
            }));

            return output;
        }
        """

        # Compile the CUDA code
        self.cuda_module = load_inline(
            name="conv_transpose2d",
            cpp_sources="",
            cuda_sources=cuda_src,
            functions=["conv_transpose2d_cuda"],
            verbose=True
        )

    def forward(self, x):
        # Call the CUDA kernel
        output = self.cuda_module.conv_transpose2d_cuda(
            x,
            self.weight,
            self.bias if self.bias is not None else torch.empty(0, device=x.device),
            self.stride,
            self.padding,
            self.output_padding,
            self.kernel_size,
            self.groups
        )
        return output