import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_3d_kernel(float* input, float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size, int stride, int padding) {
    int n = blockIdx.z * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= batch_size || c >= out_channels) return;

    for (int d = 0; d < depth_out; ++d) {
        for (int h = 0; h < height_out; ++h) {
            for (int w = 0; w < width_out; ++w) {
                float sum = 0.0f;
                int d_start = max(d * stride - padding, 0);
                int d_end = min((d + 1) * stride - padding, depth_in);
                int h_start = max(h * stride - padding, 0);
                int h_end = min((h + 1) * stride - padding, height_in);
                int w_start = max(w * stride - padding, 0);
                int w_end = min((w + 1) * stride - padding, width_in);

                for (int dd = d_start; dd < d_end; ++dd) {
                    for (int hh = h_start; hh < h_end; ++hh) {
                        for (int ww = w_start; ww < w_end; ++ww) {
                            int ii_d = dd * stride - padding + d - d_start;
                            int ii_h = hh * stride - padding + h - h_start;
                            int ii_w = ww * stride - padding + w - w_start;
                            int ii_n = n;
                            int ii_c = c;
                            sum += input[ii_n * in_channels * depth_in * height_in * width_in + ii_c * depth_in * height_in * width_in + dd * height_in * width_in + hh * width_in + ww] *
                                   weight[c * in_channels * kernel_size * kernel_size * kernel_size + (ii_c * kernel_size * kernel_size + ii_d * kernel_size + ii_h) * kernel_size + ii_w];
                        }
                    }
                }

                output[n * out_channels * depth_out * height_out * width_out + c * depth_out * height_out * width_out + d * height_out * width_out + h * width_out + w] = sum;
            }
        }
    }
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size, int stride, int padding) {
    auto out = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int block_size_x = 16;
    const int block_size_y = 16;
    const int num_blocks_x = (out_channels + block_size_x - 1) / block_size_x;
    const int num_blocks_y = (batch_size + block_size_y - 1) / block_size_y;

    conv_transpose_3d_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size_x, block_size_y)>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(),
        batch_size, in_channels, out_channels, depth_in, height_in, width_in, depth_out, height_out, width_out, kernel_size, stride, padding
    );

    return out;
}
"""

conv_transpose_3d_cpp_source = (
    "torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose_3d = load_inline(
    name="conv_transpose_3d",
    cpp_sources=conv_transpose_3d_cpp_source,
    cuda_sources=conv_transpose_3d_source,
    functions=["conv_transpose_3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for element-wise multiplication
elementwise_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_mul_kernel(float* input, float* scale, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * scale[0];
    }
}

torch::Tensor elementwise_mul_cuda(torch::Tensor input, float scale) {
    auto size = input.numel();
    auto out = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_mul_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), &scale, out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_mul_cpp_source = (
    "torch::Tensor elementwise_mul_cuda(torch::Tensor input, float scale);"
)

# Compile the inline CUDA code for element-wise multiplication
elementwise_mul = load_inline(
    name="elementwise_mul",
    cpp_sources=elementwise_mul_cpp_source,
    cuda_sources=elementwise_mul_source,
    functions=["elementwise_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose_3d
        self.scale1 = scale1
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = scale2

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_3d_cuda(x, self.weight, x.size(0), x.size(1), self.out_channels, x.size(2), x.size(3), x.size(4), x.size(2) * 2, x.size(3) * 2, x.size(4) * 2, self.kernel_size, self.stride, self.padding)
        x = self.conv_transpose.elementwise_mul_cuda(x, self.scale1)
        x = self.avg_pool(x)
        x = x + self.bias
        x = self.conv_transpose.elementwise_mul_cuda(x, self.scale2)
        return x