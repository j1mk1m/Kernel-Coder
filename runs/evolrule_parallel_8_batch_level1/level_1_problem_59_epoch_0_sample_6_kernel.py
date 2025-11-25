import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

conv3d_kernel = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* input_data,
    const float* weight_data,
    const float* bias_data,
    float* output_data,
    int N,
    int C_in,
    int D_in,
    int H_in,
    int W_in,
    int C_out,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool has_bias,
    int D_out,
    int H_out,
    int W_out
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C_out * D_out * H_out * W_out) return;

    // Compute output indices
    int w_out = index % W_out;
    int rem = index / W_out;
    int h_out = rem % H_out;
    rem /= H_out;
    int d_out = rem % D_out;
    rem /= D_out;
    int c_out = rem % C_out;
    int n = rem / C_out;

    // Compute input depth
    int input_d = d_out * stride - padding;
    if (input_d < 0 || input_d >= D_in) {
        // Apply bias if needed
        output_data[index] = has_bias ? bias_data[c_out] : 0.0f;
        return;
    }

    // Compute group
    int C_out_per_group = C_out / groups;
    int g = c_out / C_out_per_group;
    int C_in_per_group = C_in / groups;

    float sum = 0.0f;

    // Iterate over kernel spatial dimensions
    for (int k_h = 0; k_h < kernel_h; ++k_h) {
        for (int k_w = 0; k_w < kernel_w; ++k_w) {
            int input_h = h_out * stride - padding + k_h * dilation;
            int input_w = w_out * stride - padding + k_w * dilation;

            if (input_h < 0 || input_h >= H_in || input_w < 0 || input_w >= W_in) {
                continue;
            }

            // Iterate over input channels in the group
            for (int c_in_group = 0; c_in_group < C_in_per_group; ++c_in_group) {
                int c_in = g * C_in_per_group + c_in_group;

                // Input data access
                int input_offset = (
                    n * C_in * D_in * H_in * W_in
                    + c_in * D_in * H_in * W_in
                    + input_d * H_in * W_in
                    + input_h * W_in
                    + input_w
                );
                float val = input_data[input_offset];

                // Weight data access
                int weight_offset = (
                    c_out * (C_in / groups) * kernel_h * kernel_w
                    + c_in_group * kernel_h * kernel_w
                    + k_h * kernel_w
                    + k_w
                );
                float w_val = weight_data[weight_offset];

                sum += val * w_val;
            }
        }
    }

    // Add bias
    if (has_bias) {
        sum += bias_data[c_out];
    }

    output_data[index] = sum;
}

torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool has_bias,
    int D_out,
    int H_out,
    int W_out
) {
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    const int C_out = weight.size(0);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    dim3 threads(256);
    int num_elements = N * C_out * D_out * H_out * W_out;
    dim3 blocks((num_elements + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_forward", ([&] {
        conv3d_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            N, C_in, D_in, H_in, W_in,
            C_out,
            kernel_h, kernel_w,
            stride, padding, dilation,
            groups,
            has_bias,
            D_out, H_out, W_out
        );
    }));

    return output;
}
"""

conv3d_header = """
#include <torch/extension.h>
torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    bool has_bias,
    int D_out,
    int H_out,
    int W_out
);
"""

conv3d_cuda = load_inline(
    name="conv3d_cuda",
    cpp_sources=conv3d_header,
    cuda_sources=conv3d_kernel,
    functions=["conv3d_forward"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-gencode=arch=compute_70,code=sm_70"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, 1, kernel_size, kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters like PyTorch's Conv3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Calculate output dimensions
        kernel_h = kernel_w = self.kernel_size
        D_in = x.size(2)
        H_in = x.size(3)
        W_in = x.size(4)

        # Compute output spatial dimensions
        D_out = (D_in + 2 * self.padding - (1 - 1) * self.dilation - 1) // self.stride + 1
        H_out = (H_in + 2 * self.padding - (kernel_h - 1) * self.dilation - 1) // self.stride + 1
        W_out = (W_in + 2 * self.padding - (kernel_w - 1) * self.dilation - 1) // self.stride + 1

        # Ensure tensors are contiguous
        input_cont = x.contiguous()
        weight_cont = self.weight.contiguous()
        bias_cont = self.bias.contiguous() if self.bias is not None else torch.empty(0)

        # Call CUDA kernel
        output = conv3d_cuda.conv3d_forward(
            input_cont,
            weight_cont,
            bias_cont,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            D_out,
            H_out,
            W_out
        )

        return output