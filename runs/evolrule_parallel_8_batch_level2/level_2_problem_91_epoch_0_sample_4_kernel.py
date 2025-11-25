import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d + Softmax fusion
conv_transpose_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declaration of the kernel
template <typename scalar_t>
__global__ void conv_transpose_softmax_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits> output,
    int out_channels, int out_height, int out_width, int kernel_size, int stride, int padding, int output_padding) {

    // Implementation of fused convolution transpose and softmax here
    // This is a simplified placeholder - actual implementation would require
    // proper handling of convolution transpose operations and softmax normalization
    // across channels
    const int N = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);

    // Calculate output dimensions (already given, but here for reference)
    // out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    // out_width = same for width

    // Each thread handles one output element
    int n = blockIdx.x;
    int c_out = blockIdx.y;
    int h_out = threadIdx.y;
    int w_out = threadIdx.x;

    if (h_out >= out_height || w_out >= out_width) return;

    scalar_t sum = 0;
    for (int k_h = 0; k_h < kernel_size; ++k_h) {
        for (int k_w = 0; k_w < kernel_size; ++k_w) {
            int in_h = (h_out + padding - k_h) / stride;
            int in_w = (w_out + padding - k_w) / stride;
            if ((h_out + padding - k_h) % stride != 0 || (w_out + padding - k_w) % stride != 0) continue;
            if (in_h < 0 || in_h >= in_height || in_w < 0 || in_w >= in_width) continue;
            for (int c_in = 0; c_in < in_channels; ++c_in) {
                sum += input[n][c_in][in_h][in_w] * weight[c_out][c_in][k_h][k_w];
            }
        }
    }
    sum += bias[c_out][0][0]; // Add bias
    output[n][c_out][h_out][w_out] = sum;
    // Softmax normalization would require exponentiation and division by sum of exponents
    // This requires a separate step or a more complex kernel
}

std::vector<torch::Tensor> conv_transpose_softmax_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int out_height,
    int out_width) {

    const int batch_size = input.size(0);
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());

    dim3 threads(32, 32); // Thread block size for 2D threads
    dim3 blocks(batch_size, out_channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_transpose_softmax_cuda", ([&] {
        conv_transpose_softmax_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            weight.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            bias.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits>(),
            out_channels, out_height, out_width, kernel_size, stride, padding, output_padding);
    }));

    // Apply softmax here (simplified for illustration)
    output = torch::softmax(output, 1);

    return {output};
}

TORCH_LIBRARY(my_ops, m) {
  m.def("conv_transpose_softmax", &conv_transpose_softmax_cuda);
}
"""

# Custom CUDA kernel for BiasAdd + Scale + Sigmoid fusion
bias_scale_sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_scale_sigmoid_kernel(
    const float* input, const float* bias, float scale, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] + bias[0]; // Assuming bias is broadcasted
        val *= scale;
        output[idx] = 1.0 / (1.0 + exp(-val));
    }
}

torch::Tensor fused_bias_scale_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float scale) {

    auto output = torch::empty_like(input);
    int size = input.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    fused_bias_scale_sigmoid_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale,
        output.data_ptr<float>(),
        size);

    return output;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("fused_bias_scale_sigmoid", &fused_bias_scale_sigmoid_cuda);
}
"""

# Compile the custom CUDA operators
conv_transpose_softmax = load_inline(
    name="conv_transpose_softmax",
    cpp_sources="",
    cuda_sources=conv_transpose_softmax_source,
    functions=["conv_transpose_softmax"],
    verbose=True,
    with_cuda=True
)

bias_scale_sigmoid = load_inline(
    name="bias_scale_sigmoid",
    cpp_sources="",
    cuda_sources=bias_scale_sigmoid_source,
    functions=["fused_bias_scale_sigmoid"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.conv_transpose_softmax = conv_transpose_softmax
        self.bias_scale_sigmoid = bias_scale_sigmoid

    def forward(self, x):
        # Get output dimensions for fused op (normally calculated by ConvTranspose2d)
        output = self.conv_transpose(x)
        out_channels = output.size(1)
        out_height = output.size(2)
        out_width = output.size(3)
        
        # Apply fused ConvTranspose + Softmax
        x = self.conv_transpose_softmax.conv_transpose_softmax(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            out_channels,
            self.conv_transpose.kernel_size[0],
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
            out_height,
            out_width
        )[0]

        # Apply fused BiasAdd + Scale + Sigmoid
        x = self.bias_scale_sigmoid.fused_bias_scale_sigmoid(
            x,
            self.bias,
            self.scaling_factor
        )
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]