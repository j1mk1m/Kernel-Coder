import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int N, int C_in, int H_in, int W_in, int C_out, int K_h, int K_w) {
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = blockIdx.x / W_out;
    int w_out = blockIdx.x % W_out;

    int h_in_start = h_out * stride - pad;
    int w_in_start = w_out * stride - pad;

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_in = h_in_start + kh;
                int w_in = w_in_start + kw;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int i_idx = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in;
                    int k_idx = c_out * C_in * K_h * K_w + c_in * K_h * K_w + kh * K_w + kw;
                    sum += input[i_idx] * weight[k_idx];
                }
            }
        }
    }

    int o_idx = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;
    output[o_idx] = sum;
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight) {
    auto N = input.size(0);
    auto C_in = input.size(1);
    auto H_in = input.size(2);
    auto W_in = input.size(3);
    auto C_out = weight.size(0);
    auto K_h = weight.size(2);
    auto K_w = weight.size(3);
    auto stride = 1;
    auto pad = 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (H_out * W_out + block_size - 1) / block_size;

    convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N, C_in, H_in, W_in, C_out, K_h, K_w);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA kernels for convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = convolution

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight)
        x = torch.min(x, dim=1, keepdim=True)[0]  # Apply minimum operation along the channel dimension
        x = torch.tanh(x)
        x = torch.tanh(x)
        return x