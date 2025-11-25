import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void conv_transpose_3d_kernel(const float* input, const float* weight, float* output, int N, int C_in, int D_in, int H_in, int W_in, int C_out, int D_out, int H_out, int W_out, int k_d, int k_h, int k_w, int s_d, int s_h, int s_w, int p_d, int p_h, int p_w) {
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int d_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= N || c_out >= C_out || d_out >= D_out) {
        return;
    }

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int dd = 0; dd < k_d; ++dd) {
            for (int hh = 0; hh < k_h; ++hh) {
                for (int ww = 0; ww < k_w; ++ww) {
                    int d_in = d_out * s_d - p_d + dd;
                    int h_in = d_out * s_h - p_h + hh;
                    int w_in = d_out * s_w - p_w + ww;
                    if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        sum += input[n * C_in * D_in * H_in * W_in + c_in * D_in * H_in * W_in + d_in * H_in * W_in + h_in * W_in + w_in] * weight[c_out * C_in * k_d * k_h * k_w + c_in * k_d * k_h * k_w + dd * k_h * k_w + hh * k_w + ww];
                    }
                }
            }
        }
    }

    output[n * C_out * D_out * H_out * W_out + c_out * D_out * H_out * W_out + d_out * H_out * W_out] = sum;
}

torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight, int N, int C_in, int D_in, int H_in, int W_in, int C_out, int D_out, int H_out, int W_out, int k_d, int k_h, int k_w, int s_d, int s_h, int s_w, int p_d, int p_h, int p_w) {
    auto out = torch::zeros({N, C_out, D_out, H_out, W_out}, input.options());

    dim3 grid_dim((D_out + BLOCK_SIZE - 1) / BLOCK_SIZE, (H_out + BLOCK_SIZE - 1) / BLOCK_SIZE, (W_out + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    conv_transpose_3d_kernel<<<grid_dim, block_dim>>>(input.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out, k_d, k_h, k_w, s_d, s_h, s_w, p_d, p_h, p_w);

    return out;
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose_3d_cuda(torch::Tensor input, torch::Tensor weight, int N, int C_in, int D_in, int H_in, int W_in, int C_out, int D_out, int H_out, int W_out, int k_d, int k_h, int k_w, int s_d, int s_h, int s_w, int p_d, int p_h, int p_w);"
)

# Compile the inline CUDA code for 3D transposed convolution
conv_transpose = load_inline(
    name="conv_transpose",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose_3d_cuda"],
    verbose=True,
    extra_cflags=["-I/usr/local/cuda/include", "-L/usr/local/cuda/lib64", "-lcudart"],
    extra_ldflags=["-L/usr/local/cuda/lib64", "-lcudart"]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

    def forward(self, x):
        N, C_in, D_in, H_in, W_in = x.size()
        C_out, _, _, _ = self.multiplier.size()

        x = self.conv_transpose.conv_transpose_3d_cuda(x, self.multiplier.view(C_out, C_in, kernel_size, kernel_size, kernel_size), N, C_in, D_in, H_in, W_in, C_out, D_out, H_in, W_in, kernel_size, kernel_size, kernel_size, stride, stride, stride, padding, padding, padding)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = x * self.multiplier
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.max_pool3d(x, kernel_size=2)
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 16
    in_channels = 16
    out_channels = 32
    depth, height, width = 16, 32, 32
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1
    multiplier_shape = (out_channels, 1, 1, 1)

    model = ModelNew(in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape)
    inputs = get_inputs()[0].cuda()
    outputs = model(inputs)
    print(outputs.shape)