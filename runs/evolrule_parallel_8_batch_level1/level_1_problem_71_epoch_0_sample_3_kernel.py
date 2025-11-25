import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define the weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize the parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if hasattr(self, 'bias'):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Define the CUDA kernel
        kernel_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void conv_transpose2d_kernel(
            const torch::PackedTensorAccessor<scalar_t,4> input,
            const torch::PackedTensorAccessor<scalar_t,4> weight,
            const torch::PackedTensorAccessor<scalar_t,1> bias,
            torch::PackedTensorAccessor<scalar_t,4> output,
            int stride,
            int padding,
            int output_padding,
            int groups,
            int h_in,
            int w_in,
            int h_out,
            int w_out
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= output.size(0) * output.size(1) * output.size(2) * output.size(3)) return;

            int w = idx % w_out;
            int h = (idx / w_out) % h_out;
            int c_out = (idx / (h_out * w_out)) % output.size(1);
            int n = idx / (output.size(1) * h_out * w_out);

            scalar_t acc = 0;

            // Iterate over groups
            for (int g = 0; g < groups; ++g) {
                int in_group_size = input.size(1) / groups;
                int out_group_size = output.size(1) / groups;
                int in_c_start = g * in_group_size;
                int out_c_start = g * out_group_size;

                for (int c_in = in_c_start; c_in < in_c_start + in_group_size; ++c_in) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            // Compute input coordinates with adjusted indices
                            int h_in_coord = (h - kh + 2 * padding - output_padding) / stride;
                            int w_in_coord = (w - kw + 2 * padding - output_padding) / stride;

                            if (h_in_coord < 0 || h_in_coord >= h_in || 
                                w_in_coord < 0 || w_in_coord >= w_in) {
                                continue;
                            }

                            // Get group-adjusted indices
                            int w_out_idx = c_out - out_c_start;
                            int w_in_idx = c_in - in_c_start;

                            // Compute weight index
                            int weight_offset = w_out_idx * in_group_size * kernel_size * kernel_size +
                                                w_in_idx * kernel_size * kernel_size +
                                                kh * kernel_size + kw;

                            scalar_t w_val = weight[w_out_idx][w_in_idx][kh][kw];
                            scalar_t in_val = input[n][c_in][h_in_coord][w_in_coord];
                            acc += w_val * in_val;
                        }
                    }
                }
            }

            if (bias.size(0) > 0) {
                acc += bias[c_out];
            }

            output[n][c_out][h][w] = acc;
        }

        torch::Tensor conv_transpose2d_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int stride,
            int padding,
            int output_padding,
            int groups
        ) {
            auto batch_size = input.size(0);
            auto in_channels = input.size(1);
            auto h_in = input.size(2);
            auto w_in = input.size(3);
            auto kernel_size = weight.size(2);

            // Compute output dimensions
            auto h_out = (h_in - 1) * stride - 2 * padding + kernel_size + output_padding;
            auto w_out = (w_in - 1) * stride - 2 * padding + kernel_size + output_padding;

            auto output = torch::empty({batch_size, weight.size(0)*groups, h_out, w_out}, input.options());

            const int threads = 256;
            const int blocks = (output.numel() + threads - 1) / threads;

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
                auto input_acc = input.packed_accessor<scalar_t,4>();
                auto weight_acc = weight.packed_accessor<scalar_t,4>();
                auto bias_acc = bias.packed_accessor<scalar_t,1>();
                auto output_acc = output.packed_accessor<scalar_t,4>();

                conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
                    input_acc,
                    weight_acc,
                    bias_acc,
                    output_acc,
                    stride,
                    padding,
                    output_padding,
                    groups,
                    h_in,
                    w_in,
                    h_out,
                    w_out
                );
            }));

            return output;
        }
        """

        # Compile the kernel
        self.conv_transpose2d = load_inline(
            name="conv_transpose2d",
            cpp_sources="",
            cuda_sources=kernel_source,
            functions=["conv_transpose2d_cuda"],
            verbose=True
        )

    def forward(self, x):
        return self.conv_transpose2d.conv_transpose2d_cuda(
            x,
            self.weight,
            self.bias if hasattr(self, 'bias') else torch.empty(0),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )