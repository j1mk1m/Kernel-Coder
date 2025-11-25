import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_1d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int padding,
    int dilation) {

    int batch = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_pos = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_pos >= output_length) return;

    float sum = 0.0f;

    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int k = 0; k < kernel_size; k++) {
            int numerator = out_pos + padding - k * dilation;
            int in_pos = numerator / stride;

            if (in_pos < 0 || in_pos >= input_length) {
                continue;
            }

            int weight_idx = in_ch * out_channels * kernel_size + out_ch * kernel_size + k;
            int input_idx = batch * in_channels * input_length + in_ch * input_length + in_pos;
            sum += weight[weight_idx] * input[input_idx];
        }
    }

    int output_idx = batch * out_channels * output_length + out_ch * output_length + out_pos;
    output[output_idx] = sum;
}

torch::Tensor conv_transpose_1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int padding,
    int dilation) {

    const int block_size = 256;
    dim3 block(block_size);
    dim3 grid(batch_size, out_channels, (output_length + block_size - 1) / block_size);

    conv_transpose_1d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_length,
        output_length,
        stride,
        padding,
        dilation
    );

    return output;
}
"""

conv_transpose_1d = load_inline(
    name="conv_transpose_1d",
    cuda_sources=conv_transpose_1d_source,
    functions=["conv_transpose_1d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14", "--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reset_parameters()
        self.conv_transpose_1d_cuda = conv_transpose_1d

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, input_length = x.size()
        assert in_channels == self.in_channels, "Input channels mismatch"

        output_length = (input_length - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1

        output = torch.zeros(batch_size, self.out_channels, output_length, 
                            device=x.device, dtype=x.dtype)

        self.conv_transpose_1d_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            output,
            batch_size,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            input_length,
            output_length,
            self.stride,
            self.padding,
            self.dilation
        )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1)

        return output