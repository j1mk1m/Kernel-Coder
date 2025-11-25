import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const torch::PackedTensorAccessor<scalar_t, 5> input,
    const torch::PackedTensorAccessor<scalar_t, 5> weight,
    torch::PackedTensorAccessor<scalar_t, 5> output,
    int in_channels, int out_channels,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w
) {
    int batch_idx = blockIdx.x;
    int out_c = blockIdx.y;
    int out_d = threadIdx.x + blockDim.x * blockIdx.z;
    int out_h = threadIdx.y + blockDim.y * blockIdx.w;
    int out_w = threadIdx.z + blockDim.z * blockIdx.v;

    if (out_d >= output.size(2) || out_h >= output.size(3) || out_w >= output.size(4)) {
        return;
    }

    scalar_t sum = 0;
    for (int k_d = 0; k_d < kernel_d; ++k_d) {
        for (int k_h = 0; k_h < kernel_h; ++k_h) {
            for (int k_w = 0; k_w < kernel_w; ++k_w) {
                int in_d = (out_d - k_d - padding_d + output_padding_d) / stride_d;
                int in_h = (out_h - k_h - padding_h + output_padding_h) / stride_h;
                int in_w = (out_w - k_w - padding_w + output_padding_w) / stride_w;

                if (in_d >= 0 && in_d < input.size(2) &&
                    in_h >= 0 && in_h < input.size(3) &&
                    in_w >= 0 && in_w < input.size(4)) {
                    for (int in_c = 0; in_c < in_channels; ++in_c) {
                        sum += weight[out_c][in_c][k_d][k_h][k_w] * 
                               input[batch_idx][in_c][in_d][in_h][in_w];
                    }
                }
            }
        }
    }
    output[batch_idx][out_c][out_d][out_h][out_w] = sum;
}

torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, torch::Tensor weight,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);

    // Compute output dimensions
    int out_depth = (in_depth - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());

    dim3 threads(8, 8, 8);
    dim3 blocks(
        batch_size,
        out_channels,
        (out_depth + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        (out_width + threads.z - 1) / threads.z
    );

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose3d_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        conv_transpose3d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t, 5>(),
            weight.packed_accessor<scalar_t, 5>(),
            output.packed_accessor<scalar_t, 5>(),
            in_channels, out_channels,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            output_padding_d, output_padding_h, output_padding_w
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, torch::Tensor weight,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w);
"""

conv_transpose3d = load_inline(
    name="conv_transpose3d",
    cpp_sources=cpp_source,
    cuda_sources=conv_transpose3d_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cuda_cflags=['--expt-relaxed-constexpr']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1,1,1), padding: tuple = (0,0,0),
                 output_padding: tuple = (0,0,0), groups: int = 1, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weights
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *kernel_size
        ))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return conv_transpose3d.conv_transpose3d_cuda(
            x, self.weight,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2]
        )

def get_inputs():
    x = torch.randn(16, 32, 16, 32, 64).cuda()
    return [x]

def get_init_inputs():
    return [32, 16, (3,5,7), (1,1,1), (0,0,0), (0,0,0)]