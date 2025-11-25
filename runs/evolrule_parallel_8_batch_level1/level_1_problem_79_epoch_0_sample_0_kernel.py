import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for transposed 1D convolution
transposed_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void transposed_conv1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ output,
    int N, int C_in, int L_in,
    int C_out, int K,
    int stride, int padding, int dilation,
    int L_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * L_out) return;

    int n = idx / (C_out * L_out);
    int rem = idx % (C_out * L_out);
    int oc = rem / L_out;
    int o = rem % L_out;

    scalar_t sum = 0.0;
    for (int ic = 0; ic < C_in; ++ic) {
        for (int k = 0; k < K; ++k) {
            int numerator = o - k*dilation + padding;
            int input_idx = numerator / stride;
            if (input_idx * stride != numerator) continue;
            if (input_idx < 0 || input_idx >= L_in) continue;

            int weight_offset = oc * C_in * K + ic * K + k;
            int input_offset = n * C_in * L_in + ic * L_in + input_idx;
            sum += weights[weight_offset] * input[input_offset];
        }
    }

    int output_offset = n * C_out * L_out + oc * L_out + o;
    output[output_offset] = sum;
}

torch::Tensor transposed_conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    int stride,
    int padding,
    int dilation
) {
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto L_in = input.size(2);
    auto C_out = weights.size(0);
    auto K = weights.size(2);

    int L_out = (L_in - 1)*stride - 2*padding + dilation*(K - 1) + 1;

    auto output = torch::empty({N, C_out, L_out}, input.options());

    int total_threads = N * C_out * L_out;
    const int block_size = 256;
    const int num_blocks = (total_threads + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv1d_cuda", ([&] {
        transposed_conv1d_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C_in, L_in,
            C_out, K,
            stride, padding, dilation,
            L_out
        );
    }));

    return output;
}
"""

transposed_conv_header = """
torch::Tensor transposed_conv1d_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    int stride,
    int padding,
    int dilation
);
"""

transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources=transposed_conv_header,
    cuda_sources=transposed_conv_source,
    functions=["transposed_conv1d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Match PyTorch's default init

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = transposed_conv.transposed_conv1d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1)  # Add bias to each channel

        return output