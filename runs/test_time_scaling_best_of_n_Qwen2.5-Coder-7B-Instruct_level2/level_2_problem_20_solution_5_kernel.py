import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_3d_kernel(const float* input, const float* weight, float* output, int N, int C_in, int D_out, int H_out, int W_out, int C_out, int D_in, int H_in, int W_in) {
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int d_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N || c_out >= C_out || d_out >= D_out || h_out >= H_out || w_out >= W_out) {
        return;
    }

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int d_in = 0; d_in < D_in; ++d_in) {
            for (int h_in = 0; h_in < H_in; ++h_in) {
                for (int w_in = 0; w_in < W_in; ++w_in) {
                    int i_d = d_out * stride[0] - pad[0] + d_in * dilation[0];
                    int i_h = h_out * stride[1] - pad[1] + h_in * dilation[1];
                    int i_w = w_out * stride[2] - pad[2] + w_in * dilation[2];

                    if (i_d >= 0 && i_d < D_in && i_h >= 0 && i_h < H_in && i_w >= 0 && i_w < W_in) {
                        int idx_in = n * C_in * D_in * H_in * W_in + c_in * D_in * H_in * W_in + d_in * H_in * W_in + h_in * W_in + w_in;
                        int idx_weight = c_out * C_in * D_in * H_in * W_in + c_in * D_in * H_in * W_in + d_in * H_in * W_in + h_in * W_in + w_in;
                        sum += input[idx_in] * weight[idx_weight];
                    }
                }
            }
        }
    }

    int idx_out = n * C_out * D_out * H_out * W_out + c_out * D_out * H_out * W_out + d_out * H_out * W_out + h_out * W_out + w_out;
    output[idx_out] = sum;
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);
    auto C_out = weight.size(0);
    auto D_out = weight.size(2);
    auto H_out = weight.size(3);
    auto W_out = weight.size(4);

    auto output = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    dim3 block_size(16, 16, 1);
    dim3 grid_size((W_out + block_size.x - 1) / block_size.x, (H_out + block_size.y - 1) / block_size.y, (N * C_out + block_size.z - 1) / block_size.z);

    conv_transpose_3d_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N, C_in, D_out, H_out, W_out, C_out, D_in, H_in, W_in);

    return output;
}
"""

conv_transpose_3d_cpp_source = (
    "torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Define the custom CUDA kernel for element-wise multiplication
elementwise_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_mul_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

torch::Tensor elementwise_mul_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_mul_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_mul_cpp_source = (
    "torch::Tensor elementwise_mul_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for all custom operators
custom_operators = load_inline(
    name="custom_operators",
    cpp_sources=(conv_transpose_3d_cpp_source + elementwise_add_cpp_source + elementwise_mul_cpp_source),
    cuda_sources=(conv_transpose_3d_source + elementwise_add_source + elementwise_mul_source),
    functions=["conv_transpose_3d_cuda", "elementwise_add_cuda", "elementwise_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_shape = bias_shape

    def forward(self, x):
        # Perform 3D transposed convolution
        weight = torch.randn(self.out_channels, self.in_channels, *self.kernel_size).cuda()
        x = custom_operators.conv_transpose_3d_cuda(x, weight)

        # Add bias
        original_x = x.clone().detach()
        bias = torch.randn(self.out_channels, 1, 1, 1).cuda()
        x = custom_operators.elementwise_add_cuda(x, bias)

        # Residual add
        x = custom_operators.elementwise_add_cuda(x, original_x)

        # Multiplication
        x = custom_operators.elementwise_mul_cuda(x, original_x)

        # Another residual add
        x = custom_operators.elementwise_add_cuda(x, original_x)

        return x


# Example usage
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape)
inputs = get_inputs()[0].cuda()

with torch.no_grad():
    output = model_new(inputs)
print(output.shape)