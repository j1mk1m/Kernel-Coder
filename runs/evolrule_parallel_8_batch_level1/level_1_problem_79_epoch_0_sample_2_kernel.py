import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for transposed 1D convolution
conv_transpose1d_source = """
#include <torch/extension.h>

__global__ void conv_transpose1d_forward(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool has_bias,
    const float* __restrict__ bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length)
        return;

    int o_pos = idx % output_length;
    int out_c = (idx / output_length) % out_channels;
    int batch = idx / (output_length * out_channels);

    float sum = 0.0;

    for (int i_c = 0; i_c < in_channels; ++i_c) {
        for (int k = 0; k < kernel_size; ++k) {
            int temp = o_pos + padding - k * dilation;
            if (temp % stride != 0)
                continue;
            int i = temp / stride;
            if (i < 0 || i >= input_length)
                continue;
            const float w = weight[i_c * out_channels * kernel_size + out_c * kernel_size + k];
            const float in_val = input[batch * in_channels * input_length + i_c * input_length + i];
            sum += w * in_val;
        }
    }

    if (has_bias) {
        sum += bias[out_c];
    }

    output[batch * out_channels * output_length + out_c * output_length + o_pos] = sum;
}

// Wrapper function to call the kernel
torch::Tensor conv_transpose1d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool has_bias,
    torch::Tensor bias
) {
    const int threads_per_block = 256;
    const int num_blocks = (output.numel() + threads_per_block - 1) / threads_per_block;

    conv_transpose1d_forward<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation,
        has_bias,
        has_bias ? bias.data_ptr<float>() : nullptr
    );

    return output;
}
"""

conv_transpose1d_cpp_source = """
torch::Tensor conv_transpose1d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool has_bias,
    torch::Tensor bias
);
"""

# Compile the CUDA kernel
conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, input_length = x.size()
        output_length = (
            (input_length - 1) * self.stride
            - 2 * self.padding
            + self.dilation * (self.kernel_size - 1)
            + 1
        )

        output = torch.empty(
            (batch_size, self.out_channels, output_length), device=x.device
        )

        has_bias = self.bias is not None
        bias_tensor = (
            self.bias.contiguous() if has_bias else torch.empty(0, device=x.device)
        )

        output = conv_transpose1d.conv_transpose1d_cuda_forward(
            x.contiguous(),
            self.weight.contiguous(),
            output,
            batch_size,
            self.in_channels,
            self.out_channels,
            input_length,
            output_length,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            has_bias,
            bias_tensor,
        )

        return output