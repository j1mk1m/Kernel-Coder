import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused element-wise CUDA kernel
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_elementwise_kernel(
    const scalar_t* post_conv_x,
    const scalar_t* bias,
    scalar_t* out,
    int batch_size,
    int out_channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth * height * width) {
        return;
    }

    int spatial_size = depth * height * width;
    int batch = idx / (out_channels * spatial_size);
    int remainder = idx % (out_channels * spatial_size);
    int c = remainder / spatial_size;
    int spatial_idx = remainder % spatial_size;

    scalar_t bias_val = bias[c];
    scalar_t x = post_conv_x[idx];

    scalar_t step1 = x + bias_val;
    scalar_t step2 = step1 + x;
    scalar_t step3 = step2 * x;
    scalar_t step4 = step3 + x;

    out[idx] = step4;
}

std::tuple<torch::Tensor> fused_elementwise_cuda(torch::Tensor post_conv_x, torch::Tensor bias) {
    CHECK_CUDA(post_conv_x);
    CHECK_CUDA(bias);

    int batch_size = post_conv_x.size(0);
    int out_channels = post_conv_x.size(1);
    int depth = post_conv_x.size(2);
    int height = post_conv_x.size(3);
    int width = post_conv_x.size(4);
    int num_elements = batch_size * out_channels * depth * height * width;

    auto out = torch::empty_like(post_conv_x);

    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(post_conv_x.scalar_type(), "fused_elementwise_cuda", ([&] {
        fused_elementwise_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            post_conv_x.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            out_channels,
            depth,
            height,
            width
        );
    }));

    return out;
}

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_elementwise_cuda", &fused_elementwise_cuda, "Fused element-wise operations");
}
"""

# Compile the CUDA extension inline
fused_elementwise = load_inline(
    name="fused_elementwise",
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_elementwise = fused_elementwise.fused_elementwise_cuda

    def forward(self, x):
        post_conv_x = self.conv_transpose(x)
        return self.fused_elementwise(post_conv_x, self.bias)

def get_inputs():
    batch_size = 16
    in_channels = 32
    depth, height, width = 16, 32, 32
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [16, 32, 3, 2, 1, 1, (64, 1, 1, 1)]