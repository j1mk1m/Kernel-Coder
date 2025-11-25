import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_conv3d_forward(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int B, int D, int C_in, int H, int W,
    int C_out, int groups,
    int kernel_size, int stride, int padding, int dilation,
    int H_out, int W_out
) {
    const int batch = blockIdx.x;
    const int c_out = blockIdx.y;
    const int i_out = threadIdx.y;
    const int j_out = threadIdx.x;

    if (i_out >= H_out || j_out >= W_out) return;

    scalar_t acc = 0.0;

    for (int di = 0; di < kernel_size; ++di) {
        for (int dj = 0; dj < kernel_size; ++dj) {
            const int i_in = i_out * stride + di * dilation - padding;
            const int j_in = j_out * stride + dj * dilation - padding;
            if (i_in < 0 || i_in >= H || j_in < 0 || j_in >= W) continue;

            for (int g = 0; g < groups; ++g) {
                const int c_in_base = g * (C_in / groups);
                for (int c_in_group = 0; c_in_group < C_in / groups; ++c_in_group) {
                    const int c_in = c_in_base + c_in_group;
                    const int w_offset = (c_out * groups + g) * (C_in / groups) * kernel_size * kernel_size
                        + c_in_group * kernel_size * kernel_size
                        + di * kernel_size + dj;
                    const scalar_t w = weight[w_offset];

                    const int input_offset = batch * C_in * H * W
                        + c_in * H * W
                        + i_in * W + j_in;
                    acc += w * input[input_offset];
                }
            }
        }
    }

    int n = batch / D;
    int k = batch % D;
    int output_offset = n * C_out * H_out * W_out * D
        + c_out * H_out * W_out * D
        + i_out * W_out * D
        + j_out * D
        + k;
    output[output_offset] = acc;
}

at::Tensor custom_conv3d(
    at::Tensor input,
    at::Tensor weight,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    const int B = input.size(0);
    const int D = input.size(4);
    const int C_in = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int C_out = weight.size(0);
    const int kernel_size = weight.size(2); // Assuming square kernel

    // Compute output spatial dimensions
    int H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    at::Tensor output = at::zeros({B, C_out, H_out, W_out, D}, input.options());

    dim3 threads(32, 32); // threadIdx.x = j_out, threadIdx.y = i_out
    dim3 blocks(B * D, C_out); // blockIdx.x = batch (B*D), blockIdx.y = c_out

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "custom_conv3d_forward", ([&]{
        custom_conv3d_forward<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, D, C_in, H, W,
            C_out, groups,
            kernel_size, stride, padding, dilation,
            H_out, W_out
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

custom_conv3d = load_inline(
    name="custom_conv3d",
    cuda_sources=custom_conv3d_source,
    functions=["custom_conv3d"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size, 1
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Reshape weight to 4D (C_out, C_in/groups, K, K)
        weight_2d = self.weight.view(
            self.weight.size(0), self.weight.size(1), self.weight.size(2), self.weight.size(3)
        )
        output = custom_conv3d(
            x, weight_2d, self.stride, self.padding, self.dilation, self.groups
        )
        return output

# Note: get_inputs and get_init_inputs functions remain unchanged from the original.