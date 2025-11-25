import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out, int kernel_size, int stride, int padding) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch_size * out_channels * D_out * H_out * W_out) {
        int b = n / (out_channels * D_out * H_out * W_out);
        int c_out = (n % (out_channels * D_out * H_out * W_out)) / (D_out * H_out * W_out);
        int d_out = (n % (D_out * H_out * W_out)) / (H_out * W_out);
        int h_out = (n % (H_out * W_out)) / W_out;
        int w_out = n % W_out;

        float sum = 0.0f;
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                for (int k = 0; k < kernel_size; ++k) {
                    int d_in = d_out * stride - padding + i;
                    int h_in = h_out * stride - padding + j;
                    int w_in = w_out * stride - padding + k;
                    if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int idx_in = ((b * in_channels + c_out) * D_in + d_in) * H_in * W_in + (h_in * W_in + w_in);
                        int idx_weight = ((c_out * kernel_size + i) * kernel_size + j) * kernel_size + k;
                        sum += input[idx_in] * weight[idx_weight];
                    }
                }
            }
        }
        output[n] = sum;
    }
}

torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);
    auto D_out = weight.size(2);
    auto H_out = weight.size(3);
    auto W_out = weight.size(4);
    auto kernel_size = 4;
    auto stride = 2;
    auto padding = 1;

    auto output = torch::zeros({batch_size, out_channels, D_out, H_out, W_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * D_out * H_out * W_out + block_size - 1) / block_size;

    conv_transpose_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, D_in, H_in, W_in, D_out, H_out, W_out, kernel_size, stride, padding);

    return output;
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_cuda(x, self.weight)
        x = self.layer_norm(x)
        x = torch.nn.functional.gelu(x)
        x = x * self.scaling_factor
        return x

batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]