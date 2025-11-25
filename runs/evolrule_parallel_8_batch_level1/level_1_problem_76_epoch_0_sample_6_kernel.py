import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel source code
conv1d_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void custom_conv1d_forward(
    const float* input,
    const float* weight,
    float* output,
    int B,
    int C_in,
    int C_out,
    int L,
    int L_out,
    int kernel_size,
    int stride,
    int dilation
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * C_out * L_out) return;

    int b = tid / (C_out * L_out);
    int rem = tid % (C_out * L_out);
    int oc = rem / L_out;
    int ol = rem % L_out;

    int start = ol * stride;
    int pos0 = start;
    int pos1 = start + dilation;
    int pos2 = start + 2 * dilation;

    float sum = 0.0f;

    for (int ic = 0; ic < C_in; ++ic) {
        const int weight_offset = oc * C_in * kernel_size + ic * kernel_size;
        float w0 = weight[weight_offset + 0];
        float w1 = weight[weight_offset + 1];
        float w2 = weight[weight_offset + 2];

        const int input_base = b * C_in * L + ic * L;
        float in0 = input[input_base + pos0];
        float in1 = input[input_base + pos1];
        float in2 = input[input_base + pos2];

        sum += w0 * in0 + w1 * in1 + w2 * in2;
    }

    int output_offset = b * C_out * L_out + oc * L_out + ol;
    output[output_offset] = sum;
}

torch::Tensor custom_conv1d(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int dilation,
    int kernel_size
) {
    int B = input.size(0);
    int C_in = input.size(1);
    int C_out = weight.size(0);
    int L = input.size(2);

    int L_out = (L - (dilation * (kernel_size - 1) + 1)) / stride + 1;

    auto output = torch::empty({B, C_out, L_out}, input.options());

    int total_threads = B * C_out * L_out;
    int block_size = 1024;
    int grid_size = (total_threads + block_size - 1) / block_size;

    custom_conv1d_forward<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, C_out, L, L_out, kernel_size, stride, dilation
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}
"""

conv1d_cpp_source = (
    "torch::Tensor custom_conv1d(torch::Tensor input, torch::Tensor weight, int stride, int dilation, int kernel_size);"
)

# Load the CUDA extension
custom_conv1d = load_inline(
    name="custom_conv1d",
    cpp_sources=conv1d_cpp_source,
    cuda_sources=conv1d_kernel_source,
    functions=["custom_conv1d"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.bias = bias

        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters like nn.Conv1d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the custom convolution
        out = custom_conv1d(x, self.weight, self.stride, self.dilation, self.kernel_size)
        # Add bias if present
        if self.bias is not None:
            out += self.bias.view(1, -1, 1)
        return out