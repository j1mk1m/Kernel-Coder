import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose3d
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Define the kernel dimensions
constexpr int BLOCK_SIZE = 128;

// Kernel for transposed 3D convolution
template <typename scalar_t>
__global__ void conv_transpose_3d_kernel(
    const torch::PackedTensorAccessor<scalar_t,5> input,
    const torch::PackedTensorAccessor<scalar_t,5> weight,
    torch::PackedTensorAccessor<scalar_t,5> output,
    int out_channels, int in_channels,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w) {

    // Calculate output indices based on thread/block IDs
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (d >= output.size(2) || h >= output.size(3) || w >= output.size(4)) {
        return;
    }

    // Compute input coordinates
    int od = d;
    int oh = h;
    int ow = w;

    for (int b = 0; b < output.size(0); ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            scalar_t sum = 0;
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kd = 0; kd < kernel_depth; ++kd) {
                    for (int kh = 0; kh < kernel_height; ++kh) {
                        for (int kw = 0; kw < kernel_width; ++kw) {
                            // Compute input spatial coordinates
                            int id = (od - kd + 2 * padding_d - kernel_depth + 1) / stride_d;
                            int ih = (oh - kh + 2 * padding_h - kernel_height + 1) / stride_h;
                            int iw = (ow - kw + 2 * padding_w - kernel_width + 1) / stride_w;

                            // Check if input coordinates are valid
                            if (id >= 0 && id < input.size(2) &&
                                ih >= 0 && ih < input.size(3) &&
                                iw >= 0 && iw < input.size(4)) {
                                sum += input[b][ic][id][ih][iw] *
                                       weight[oc][ic][kd][kh][kw];
                            }
                        }
                    }
                }
            }
            output[b][oc][od][oh][ow] = sum;
        }
    }
}

// Wrapper function
at::Tensor conv_transpose_3d_cuda(
    at::Tensor input,
    at::Tensor weight,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w) {

    auto output_size = torch::IntArrayRef({
        input.size(0),
        weight.size(0),
        (input.size(2) - 1) * stride_d - 2 * padding_d + weight.size(2),
        (input.size(3) - 1) * stride_h - 2 * padding_h + weight.size(3),
        (input.size(4) - 1) * stride_w - 2 * padding_w + weight.size(4)
    });

    auto output = at::empty(output_size, input.options());

    dim3 threads(BLOCK_SIZE, 1, 1);
    dim3 blocks(
        (output.size(4) + threads.x - 1) / threads.x,
        (output.size(3) + threads.y - 1) / threads.y,
        (output.size(2) + threads.z - 1) / threads.z);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_3d_cuda", ([&] {
        conv_transpose_3d_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5>(),
            weight.packed_accessor<scalar_t,5>(),
            output.packed_accessor<scalar_t,5>(),
            weight.size(0), weight.size(1),
            weight.size(2), weight.size(3), weight.size(4),
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose_3d_cpp_source = """
at::Tensor conv_transpose_3d_cuda(
    at::Tensor input,
    at::Tensor weight,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w);
"""

# Compile the custom kernel
conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cuda_sources=conv_transpose_3d_source,
    cpp_sources=conv_transpose_3d_cpp_source,
    functions=["conv_transpose_3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        # Custom transposed convolution parameters
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        
        # Other PyTorch operations remain unchanged
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        
        # Reference to the custom kernel
        self.conv_transpose_3d_cuda = conv_transpose_3d.conv_transpose_3d_cuda

    def forward(self, x):
        # Custom conv transpose using CUDA kernel
        x = self.conv_transpose_3d_cuda(
            x,
            self.weight,
            self.stride, self.stride, self.stride,  # Assuming same stride in all dimensions
            self.padding, self.padding, self.padding  # Same padding
        )
        
        # Remaining operations using native PyTorch
        x = x * self.scale1
        x = self.avg_pool(x)
        x = x + self.bias
        x = x * self.scale2
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]

# Configuration parameters (same as original)
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)