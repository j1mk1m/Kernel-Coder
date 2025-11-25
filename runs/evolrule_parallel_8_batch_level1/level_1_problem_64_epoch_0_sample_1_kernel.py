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

        # Initialize weights similar to PyTorch's ConvTranspose1d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.bias_param = None

        # Precompute rotated kernel
        self.register_buffer('rotated_weight', torch.flip(self.weight.data, [2]))

        # Define the CUDA kernel for transposed convolution
        transposed_conv_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void transposed_conv1d_forward(
            const scalar_t* input,
            const scalar_t* weight,
            scalar_t* output,
            int batch_size,
            int in_channels,
            int out_channels,
            int kernel_size,
            int input_length,
            int output_length,
            int stride,
            int padding,
            int output_padding,
            int groups
        ) {
            // Calculate thread and block indices
            int batch = blockIdx.x;
            int out_channel = blockIdx.y;
            int out_pos = threadIdx.x + blockDim.x * blockIdx.z;

            if (out_pos >= output_length) return;

            // Compute input position based on output position
            int effective_output_pos = out_pos - output_padding;
            int in_pos = (effective_output_pos - padding) / stride;

            // Check bounds
            if (in_pos < 0 || in_pos >= input_length) return;

            scalar_t acc = 0;
            for (int k = 0; k < kernel_size; ++k) {
                int kernel_pos = k;
                int in_channel_base = (out_channel / (out_channels / groups)) * (in_channels / groups);
                for (int g = 0; g < groups; ++g) {
                    int in_channel = in_channel_base + (g * (in_channels / groups));
                    int input_idx = batch * in_channels * input_length +
                                   in_channel * input_length +
                                   in_pos + kernel_pos;
                    int weight_idx = (out_channel % (out_channels / groups)) * kernel_size + kernel_pos;
                    acc += input[input_idx] * weight[weight_idx];
                }
            }

            // Store result
            int output_offset = batch * out_channels * output_length +
                               out_channel * output_length +
                               out_pos;
            atomicAdd(&output[output_offset], acc);
        }

        torch::Tensor transposed_conv1d_forward_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            int batch_size,
            int in_channels,
            int out_channels,
            int kernel_size,
            int input_length,
            int output_length,
            int stride,
            int padding,
            int output_padding,
            int groups
        ) {
            auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());

            int threads_per_block = 256;
            dim3 blocks(batch_size, out_channels, (output_length + threads_per_block - 1) / threads_per_block);
            dim3 threads(threads_per_block);

            AT_DISPATCH_FLOATING_TYPES(input.type(), "transposed_conv1d_forward", ([&] {
                transposed_conv1d_forward<scalar_t><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    kernel_size,
                    input_length,
                    output_length,
                    stride,
                    padding,
                    output_padding,
                    groups
                );
            }));

            cudaDeviceSynchronize();
            return output;
        }
        """

        # Compile the CUDA kernel
        self.transposed_conv = load_inline(
            name="transposed_conv",
            cpp_sources="",
            cuda_sources=transposed_conv_source,
            functions=["transposed_conv1d_forward_cuda"],
            verbose=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate output length
        batch_size, _, input_length = x.size()
        output_length = (input_length - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Execute the CUDA kernel
        output = self.transposed_conv.transposed_conv1d_forward_cuda(
            x.cuda(),
            self.rotated_weight.cuda(),
            batch_size,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            input_length,
            output_length,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )

        # Add bias if present
        if self.bias_param is not None:
            output += self.bias_param.view(1, -1, 1).cuda()

        return output