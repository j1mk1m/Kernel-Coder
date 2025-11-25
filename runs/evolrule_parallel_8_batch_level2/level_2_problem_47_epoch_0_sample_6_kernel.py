import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused Conv3D + Mish + Tanh kernel
fused_conv_mish_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;      \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void fused_conv_mish_tanh_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    const int kernel_size, const int stride, const int padding) {

    const int n = output.size(0);
    const int channels_out = output.size(1);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    CUDA_1D_KERNEL_LOOP(index, n * channels_out * out_h * out_w) {
        int w = index % out_w;
        int h = (index / out_w) % out_h;
        int c_out = (index / (out_w * out_h)) % channels_out;
        int n = index / (out_w * out_h * channels_out);

        scalar_t sum = 0;
        for (int k = 0; k < kernel_size; ++k) {
            for (int kk = 0; kk < kernel_size; ++kk) {
                for (int d = 0; d < kernel_size; ++d) {
                    for (int c_in = 0; c_in < input.size(1); ++c_in) {
                        int x = w * stride - padding + kk;
                        int y = h * stride - padding + k;
                        int z = n * stride - padding + d; // Assuming depth dimension
                        if (x >= 0 && x < input.size(3) && y >= 0 && y < input.size(2) && z >=0 && z < input.size(1)) {
                            sum += input[n][z][y][x] * weight[c_out][c_in][d][k][kk];
                        }
                    }
                }
            }
        }

        // Apply Mish activation: x * tanh(softplus(x))
        scalar_t softplus = std::log(1 + std::exp(sum));
        scalar_t mish = sum * std::tanh(softplus);

        // Apply Tanh activation
        output[n][c_out][h][w] = std::tanh(mish);
    }
}

torch::Tensor fused_conv_mish_tanh_cuda(torch::Tensor input,
                                       torch::Tensor weight,
                                       int kernel_size,
                                       int stride,
                                       int padding) {
    // Get output dimensions
    const int batch = input.size(0);
    const int channels_in = input.size(1);
    const int depth_in = input.size(2);
    const int height_in = input.size(3);
    const int width_in = input.size(4);

    const int channels_out = weight.size(0);
    const int depth_out = (depth_in + 2 * padding - kernel_size) / stride + 1;
    const int height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    const int width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::zeros({batch, channels_out, depth_out, height_out, width_out}, output_options);

    int threads = 256;
    int elements = batch * channels_out * depth_out * height_out * width_out;
    int blocks = (elements + threads - 1) / threads;

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv_mish_tanh_cuda", ([&] {
        fused_conv_mish_tanh_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            kernel_size, stride, padding);
    }));

    return output;
}
"""

fused_conv_mish_tanh_cpp = """
torch::Tensor fused_conv_mish_tanh_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding);
"""

# Compile the fused kernel
fused_conv_mish_tanh = load_inline(
    name="fused_conv_mish_tanh",
    cpp_sources=fused_conv_mish_tanh_cpp,
    cuda_sources=fused_conv_mish_tanh_source,
    functions=["fused_conv_mish_tanh_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=['-lineinfo', '-arch=sm_75']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))

    def forward(self, x):
        return fused_conv_mish_tanh.fused_conv_mish_tanh_cuda(
            x, self.weight, self.kernel_size, self.stride, self.padding
        )

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]