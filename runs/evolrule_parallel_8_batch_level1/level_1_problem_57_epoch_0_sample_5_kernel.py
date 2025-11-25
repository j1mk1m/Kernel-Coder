import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weights similar to PyTorch's ConvTranspose2d
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Compile the CUDA kernel
        self.conv_transpose2d = load_inline(
            name="conv_transpose2d",
            cpp_sources=f"""
                torch::Tensor conv_transpose2d(torch::Tensor input, torch::Tensor weight);
            """,
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <vector>

                template <typename scalar_t>
                __global__ void conv_transpose2d_kernel(
                    const scalar_t* __restrict__ input,
                    const scalar_t* __restrict__ weight,
                    scalar_t* __restrict__ output,
                    int batch_size,
                    int in_channels,
                    int out_channels,
                    int kernel_size,
                    int stride,
                    int padding,
                    int output_padding,
                    int groups,
                    int input_height,
                    int input_width,
                    int output_height,
                    int output_width
                ) {{
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx >= batch_size * out_channels * output_height * output_width)
                        return;

                    int w_out = idx % output_width;
                    int h_out = (idx / output_width) % output_height;
                    int c_out = (idx / (output_height * output_width)) % out_channels;
                    int n = idx / (out_channels * output_height * output_width);

                    scalar_t sum = 0.0;

                    int out_channels_per_group = out_channels / groups;
                    int in_channels_per_group = in_channels / groups;

                    int group = c_out / out_channels_per_group;
                    int c_out_in_group = c_out % out_channels_per_group;
                    int c_in_base = group * in_channels_per_group;

                    for (int c_in = 0; c_in < in_channels_per_group; ++c_in) {{
                        int c_in_global = c_in_base + c_in;
                        for (int kh = 0; kh < kernel_size; ++kh) {{
                            for (int kw = 0; kw < kernel_size; ++kw) {{
                                // Compute input coordinates
                                int h_in = (h_out - kh + 2 * padding - output_padding) / stride;
                                int w_in = (w_out - kw + 2 * padding) / stride;

                                if (h_in >= 0 && h_in < input_height &&
                                    w_in >= 0 && w_in < input_width) {{
                                    // Compute weight index
                                    int weight_offset = c_in * out_channels_per_group * kernel_size * kernel_size +
                                                        c_out_in_group * kernel_size * kernel_size +
                                                        kh * kernel_size + kw;

                                    // Access input and weight
                                    int input_offset = n * in_channels * input_height * input_width +
                                                        c_in_global * input_height * input_width +
                                                        h_in * input_width + w_in;

                                    sum += input[input_offset] * weight[weight_offset];
                                }}
                            }}
                        }}
                    }}

                    // Write output
                    int output_offset = n * out_channels * output_height * output_width +
                                        c_out * output_height * output_width +
                                        h_out * output_width + w_out;
                    output[output_offset] = sum;
                }}

                at::Tensor conv_transpose2d_cuda(
                    at::Tensor input,
                    at::Tensor weight
                ) {{
                    auto batch_size = input.size(0);
                    auto in_channels = input.size(1);
                    auto input_height = input.size(2);
                    auto input_width = input.size(3);

                    auto out_channels = weight.size(1) * weight.size(0); // Assuming groups=1 for simplicity
                    auto kernel_size = weight.size(2);

                    // Compute output dimensions
                    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
                    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

                    auto output = at::zeros({{batch_size, out_channels, output_height, output_width}}, input.options());

                    const int threads_per_block = 256;
                    const int blocks = (output.numel() + threads_per_block - 1) / threads_per_block;

                    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose2d_cuda", ([&] {{
                        conv_transpose2d_kernel<scalar_t><<<blocks, threads_per_block>>>(
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
                            input_height,
                            input_width,
                            output_height,
                            output_width
                        );
                    }}));

                    return output;
                }}

                torch::Tensor conv_transpose2d(torch::Tensor input, torch::Tensor weight) {{
                    CHECK_INPUT(input);
                    CHECK_INPUT(weight);
                    return conv_transpose2d_cuda(input, weight);
                }}
            """,
            extra_cuda_cflags=['-lineinfo'],
            extra_cflags=[''],
            extra_ldflags=[''],
            verbose=True
        )

    def forward(self, x):
        return self.conv_transpose2d.conv_transpose2d(x, self.weight)