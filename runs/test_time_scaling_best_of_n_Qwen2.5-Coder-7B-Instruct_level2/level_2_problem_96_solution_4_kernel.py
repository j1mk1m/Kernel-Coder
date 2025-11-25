import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 3D convolution and scaling
conv_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_scale_kernel(float* input, const float* weight, float* output, int n, int c_in, int d_in, int h_in, int w_in, int c_out, int d_out, int h_out, int w_out, float scale) {
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_idx >= n) return;

    for (int c = 0; c < c_out; ++c) {
        for (int d = 0; d < d_out; ++d) {
            for (int h = 0; h < h_out; ++h) {
                for (int w = 0; w < w_out; ++w) {
                    int o_idx = n_idx * c_out * d_out * h_out * w_out + c * d_out * h_out * w_out + d * h_out * w_out + h * w_out + w;
                    float sum = 0.0f;
                    for (int cd = 0; cd < c_in; ++cd) {
                        for (int dh = 0; dh < d_in; ++dh) {
                            for (int dw = 0; dw < w_in; ++dw) {
                                int i_idx = n_idx * c_in * d_in * h_in * w_in + cd * d_in * h_in * w_in + dh * h_in * w_in + dw * w_in + (d - dh) * h_in * w_in + (h - dh) * w_in + (w - dw);
                                sum += input[i_idx] * weight[c * c_in * d_in * h_in * w_in + cd * d_in * h_in * w_in + dh * h_in * w_in + dw * w_in];
                            }
                        }
                    }
                    output[o_idx] = sum * scale;
                }
            }
        }
    }
}

torch::Tensor conv_scale_cuda(torch::Tensor input, torch::Tensor weight, float scale) {
    auto n = input.size(0);
    auto c_in = input.size(1);
    auto d_in = input.size(2);
    auto h_in = input.size(3);
    auto w_in = input.size(4);
    auto c_out = weight.size(0);
    auto d_out = weight.size(1);
    auto h_out = weight.size(2);
    auto w_out = weight.size(3);

    auto output = torch::zeros({n, c_out, d_out, h_out, w_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (n * c_out * d_out * h_out * w_out + block_size - 1) / block_size;

    conv_scale_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), n, c_in, d_in, h_in, w_in, c_out, d_out, h_out, w_out, scale);

    return output;
}
"""

conv_scale_cpp_source = (
    "torch::Tensor conv_scale_cuda(torch::Tensor input, torch::Tensor weight, float scale);"
)

# Compile the inline CUDA code for transposed 3D convolution and scaling
conv_scale = load_inline(
    name="conv_scale",
    cpp_sources=conv_scale_cpp_source,
    cuda_sources=conv_scale_source,
    functions=["conv_scale_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for max pooling and global average pooling
pool_avg_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void pool_avg_kernel(const float* input, float* output, int n, int c, int d_in, int h_in, int w_in, int d_out, int h_out, int w_out) {
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_idx >= n) return;

    for (int c_idx = 0; c_idx < c; ++c_idx) {
        for (int d = 0; d < d_out; ++d) {
            for (int h = 0; h < h_out; ++h) {
                for (int w = 0; w < w_out; ++w) {
                    int o_idx = n_idx * c * d_out * h_out * w_out + c_idx * d_out * h_out * w_out + d * h_out * w_out + h * w_out + w;
                    float sum = 0.0f;
                    for (int dd = 0; dd < d_in; ++dd) {
                        for (int hh = 0; hh < h_in; ++hh) {
                            for (int ww = 0; ww < w_in; ++ww) {
                                int i_idx = n_idx * c * d_in * h_in * w_in + c_idx * d_in * h_in * w_in + dd * h_in * w_in + hh * w_in + ww;
                                sum += input[i_idx];
                            }
                        }
                    }
                    output[o_idx] = sum / (d_in * h_in * w_in);
                }
            }
        }
    }
}

torch::Tensor pool_avg_cuda(torch::Tensor input) {
    auto n = input.size(0);
    auto c = input.size(1);
    auto d_in = input.size(2);
    auto h_in = input.size(3);
    auto w_in = input.size(4);
    auto d_out = d_in;
    auto h_out = h_in;
    auto w_out = w_in;

    auto output = torch::zeros({n, c, d_out, h_out, w_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (n * c * d_out * h_out * w_out + block_size - 1) / block_size;

    pool_avg_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), n, c, d_in, h_in, w_in, d_out, h_out, w_out);

    return output;
}
"""

pool_avg_cpp_source = (
    "torch::Tensor pool_avg_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for max pooling and global average pooling
pool_avg = load_inline(
    name="pool_avg",
    cpp_sources=pool_avg_cpp_source,
    cuda_sources=pool_avg_source,
    functions=["pool_avg_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_scale = conv_scale
        self.pool_avg = pool_avg

    def forward(self, x):
        x = self.conv_scale.conv_scale_cuda(x, torch.ones_like(x), scale)
        x = self.pool_avg.pool_avg_cuda(x)
        x = torch.clamp(x, min=0, max=1)
        return x


# Example usage
model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size)
inputs = get_inputs()
output = model_new(inputs[0])
print(output.shape)  # Should print torch.Size([128, 16, 1, 1, 1])