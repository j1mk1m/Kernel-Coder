import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void conv3d_kernel(
    const float* input, const float* kernel, float* output,
    int N, int C_in, int H, int W, int D,
    int C_out, int C_per_group, int Kh, int Kw,
    int stride, int padding, int dilation,
    int H_out, int W_out, int D_out, int groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N * C_out * H_out * W_out * D_out) return;

    int d_out = idx / (N * C_out * H_out * W_out);
    int rem = idx % (N * C_out * H_out * W_out);
    int n = rem / (C_out * H_out * W_out);
    rem %= (C_out * H_out * W_out);
    int co = rem / (H_out * W_out);
    rem %= (H_out * W_out);
    int h_out = rem / W_out;
    int w_out = rem % W_out;

    int co_group = co / (C_out / groups);
    int co_in_group = co % (C_out / groups);

    float sum = 0.0f;
    for (int kh = 0; kh < Kh; ++kh) {
        for (int kw = 0; kw < Kw; ++kw) {
            // Compute input indices with padding and stride/dilation
            int h_in = h_out * stride + kh * dilation - padding;
            int w_in = w_out * stride + kw * dilation - padding;
            int d_in = d_out * stride + 0 * dilation - padding; // kernel depth is 1

            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W || d_in <0 || d_in >= D) {
                continue;
            }

            for (int c_in_group = 0; c_in_group < C_per_group; ++c_in_group) {
                int c_in = co_group * C_per_group + c_in_group;

                // Calculate input offset
                int input_offset = n * C_in * H * W * D +
                    c_in * H * W * D +
                    h_in * W * D +
                    w_in * D +
                    d_in;

                // Calculate kernel offset
                int kernel_offset = (co_group * (C_out / groups) + co_in_group) * C_per_group * Kh * Kw +
                    c_in_group * Kh * Kw +
                    kh * Kw +
                    kw;

                sum += input[input_offset] * kernel[kernel_offset];
            }
        }
    }

    // Output index
    int output_offset = n * C_out * H_out * W_out * D_out +
        co * H_out * W_out * D_out +
        h_out * W_out * D_out +
        w_out * D_out +
        d_out;

    output[output_offset] = sum;
}

torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Get input dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int D = input.size(4);

    // Kernel dimensions
    int C_out = kernel.size(0);
    int C_per_group = kernel.size(1);
    int Kh = kernel.size(2);
    int Kw = kernel.size(3);

    // Compute output dimensions with dilation
    int effective_Kh = Kh + (Kh - 1) * (dilation - 1);
    int effective_Kw = Kw + (Kw - 1) * (dilation - 1);
    int H_out = (H + 2 * padding - effective_Kh) / stride + 1;
    int W_out = (W + 2 * padding - effective_Kw) / stride + 1;
    int D_out = (D + 2 * padding - 1) / stride + 1; // kernel's depth is 1

    // Output tensor
    auto output = torch::zeros({N, C_out, H_out, W_out, D_out}, input.options());

    // Launch kernel
    int total_threads = N * C_out * H_out * W_out * D_out;
    int blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    conv3d_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W, D,
        C_out, C_per_group, Kh, Kw,
        stride, padding, dilation,
        H_out, W_out, D_out, groups
    );

    return output;
}
"""

conv3d_cuda_cpp = """
torch::Tensor conv3d_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    int stride,
    int padding,
    int dilation,
    int groups);
"""

conv3d_cuda = load_inline(
    name="conv3d_cuda",
    cpp_sources=conv3d_cpp,
    cuda_sources=conv3d_cuda_source,
    functions=["conv3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Create the kernel weight parameter with shape (out_channels, in_channels/groups, kernel_size, kernel_size, 1)
        kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size, 1)
        self.weight = nn.Parameter(torch.empty(kernel_shape))
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv3d_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )