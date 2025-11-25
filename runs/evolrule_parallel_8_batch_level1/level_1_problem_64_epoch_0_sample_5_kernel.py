import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose_1d_kernel(
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
    int input_length,
    int output_length,
    int out_channels_per_group,
    int in_channels_per_group) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    int b = idx / (out_channels * output_length);
    int rem = idx % (out_channels * output_length);
    int c_out = rem / output_length;
    int t_out = rem % output_length;

    int group = c_out / out_channels_per_group;
    int c_in_base = group * in_channels_per_group;
    int c_out_in_group = c_out % out_channels_per_group;

    scalar_t acc = 0;

    for (int k = 0; k < kernel_size; ++k) {
        int input_t = (t_out + padding - k) / stride;

        if (input_t >=0 && input_t < input_length) {
            for (int c_in = 0; c_in < in_channels_per_group; ++c_in) {
                int c_in_full = c_in_base + c_in;

                // Access input: batch, c_in, input_t
                int input_offset = b * in_channels * input_length + c_in_full * input_length + input_t;
                scalar_t in_val = input[input_offset];

                // Weight: (c_in, c_out_in_group, k)
                int weight_offset = (c_in * out_channels_per_group + c_out_in_group) * kernel_size + k;
                scalar_t w = weight[weight_offset];

                acc += in_val * w;
            }
        }
    }

    if (bias) {
        acc += bias[c_out];
    }

    // Write to output: batch, c_out, t_out
    int output_offset = b * out_channels * output_length + c_out * output_length + t_out;
    output[output_offset] = acc;
}

torch::Tensor conv_transpose_1d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    torch::Tensor bias,
                                    int stride,
                                    int padding,
                                    int output_padding,
                                    int groups) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int length = input.size(2);
    int out_channels = weight.size(1) * groups;
    int kernel_size = weight.size(2);
    int out_channels_per_group = out_channels / groups;
    int in_channels_per_group = in_channels / groups;
    int output_length = (length - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::empty({batch_size, out_channels, output_length},
                              input.options());

    int num_elements = batch_size * out_channels * output_length;
    int threads_per_block = 256;
    dim3 blocks((num_elements + threads_per_block - 1) / threads_per_block);
    dim3 threads(threads_per_block);

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose_1d_cuda", ([&] {
        using scalar_t = scalar_type;
        conv_transpose_1d_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
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
            length,
            output_length,
            out_channels_per_group,
            in_channels_per_group
        );
    }));

    cudaDeviceSynchronize();

    return output;
}
"""

conv_transpose_1d_cpp = """
torch::Tensor conv_transpose_1d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    torch::Tensor bias,
                                    int stride,
                                    int padding,
                                    int output_padding,
                                    int groups);
"""

conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_1d_cpp,
    cuda_sources=conv_transpose_1d_source,
    functions=["conv_transpose_1d_cuda"],
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

        # Initialize parameters
        weight_shape = (in_channels, out_channels // groups, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters like PyTorch's ConvTranspose1d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, device=x.device)
        return conv_transpose.conv_transpose_1d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            bias_tensor.contiguous(),
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )