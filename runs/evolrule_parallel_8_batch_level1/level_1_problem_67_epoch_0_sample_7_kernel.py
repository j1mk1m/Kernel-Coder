import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 1D convolution
custom_convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;    \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void custom_convolution_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weight,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const bool has_bias,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> bias) {

    CUDA_1D_KERNEL_LOOP(output_index, batch_size * out_channels) {
        int n = output_index / out_channels;
        int c_out = output_index % out_channels;
        
        int in_group = c_out / groups;
        int group_id = c_out % groups;

        int c_in = group_id * (in_channels / groups);
        
        const int output_length = output.size(2);
        for (int pos = 0; pos < output_length; ++pos) {
            scalar_t sum = 0;
            for (int k = 0; k < kernel_size; ++k) {
                int input_pos = pos * stride - padding + k * dilation;
                if (input_pos >= 0 && input_pos < input_length) {
                    for (int c = 0; c < (in_channels / groups); ++c) {
                        sum += input[n][c_in + c][input_pos] * weight[c_out][c][k];
                    }
                }
            }
            if (has_bias) {
                sum += bias[c_out];
            }
            output[n][c_out][pos] = sum;
        }
    }
}

torch::Tensor custom_convolution(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool has_bias) {

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);

    const int output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, out_channels, output_length}, output_options);

    dim3 blocks;
    dim3 threads;

    int total_elements = batch_size * out_channels;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    auto input_acc = input.packed_accessor<float,3,torch::RestrictPtrTraits>();
    auto weight_acc = weight.packed_accessor<float,3,torch::RestrictPtrTraits>();
    auto output_acc = output.packed_accessor<float,3,torch::RestrictPtrTraits>();
    auto bias_acc = bias.packed_accessor<float,1,torch::RestrictPtrTraits>();

    custom_convolution_kernel<float><<<num_blocks, threads_per_block>>>(
        input_acc,
        weight_acc,
        output_acc,
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        has_bias,
        bias_acc
    );

    cudaDeviceSynchronize();
    return output;
}
"""

custom_convolution_cpp_source = """
torch::Tensor custom_convolution(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool has_bias);
"""

# Compile the inline CUDA code for convolution
custom_convolution_cuda = load_inline(
    name="custom_convolution_cuda",
    cpp_sources=custom_convolution_cpp_source,
    cuda_sources=custom_convolution_source,
    functions=["custom_convolution"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias similar to PyTorch's Conv1d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

        # Assign the custom CUDA function
        self.custom_conv = custom_convolution_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if bias is present
        has_bias = self.bias is not None

        # Run the custom convolution
        return self.custom_conv(
            x,
            self.weight,
            self.bias if has_bias else torch.zeros(1),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            has_bias
        )