import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv_transpose1d_kernel(
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
    int dilation,
    bool has_bias,
    const scalar_t* bias,
    int output_offset,
    int input_offset,
    int weight_offset
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= output_length) return;

    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            scalar_t sum = 0;
            for (int k = 0; k < kernel_size; ++k) {
                int kernel_element = kernel_size - 1 - k;
                int numerator = o + padding + kernel_element * dilation;
                int i = (numerator / stride) - padding;
                if (i >= 0 && i < input_length) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        scalar_t input_val = input[
                            b * in_channels * input_length +
                            ic * input_length + i
                        ];
                        scalar_t weight_val = weight[
                            ic * out_channels * kernel_size +
                            oc * kernel_size + kernel_element
                        ];
                        sum += input_val * weight_val;
                    }
                }
            }
            if (has_bias) {
                sum += bias[oc];
            }
            output[
                b * out_channels * output_length +
                oc * output_length + o
            ] = sum;
        }
    }
}

at::Tensor conv_transpose1d_cuda(
    const at::Tensor &input,
    const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias,
    int stride,
    int padding,
    int dilation
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(1);
    auto kernel_size = weight.size(2);
    auto input_length = input.size(2);

    int output_length = (input_length - 1) * stride - 2 * padding +
                        dilation * (kernel_size - 1) + 1;
    auto output = at::empty({batch_size, out_channels, output_length}, input.options());

    int block_size = 256;
    int num_blocks = (output_length + block_size - 1) / block_size;

    dim3 blocks(num_blocks);
    dim3 threads(block_size);

    bool has_bias_flag = bias.has_value();
    const auto* bias_ptr = has_bias_flag ? bias.value().data_ptr<scalar_t>() : nullptr;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose1d_cuda", ([&] {
        conv_transpose1d_kernel<scalar_t><<<blocks, threads>>>(
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
            dilation,
            has_bias_flag,
            bias_ptr,
            0, 0, 0
        );
    }));

    return output;
}
"""

cpp_sources = """
at::Tensor conv_transpose1d_cuda(
    const at::Tensor &input,
    const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias,
    int stride,
    int padding,
    int dilation
);
"""

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)

        self.reset_parameters()

        self.conv_transpose_cuda = load_inline(
            name="conv_transpose_cuda",
            cpp_sources=cpp_sources,
            cuda_sources=conv_transpose1d_source,
            functions=["conv_transpose1d_cuda"],
            verbose=True,
            extra_cflags=["-g", "-O3"],
            extra_cuda_cflags=["-g", "--use_fast_math", "--expt-relaxed-constexpr"]
        )

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        bias = self.bias if self.bias is not None else None
        return self.conv_transpose_cuda.conv_transpose1d_cuda(
            x,
            self.weight,
            bias,
            self.stride,
            self.padding,
            self.dilation
        )