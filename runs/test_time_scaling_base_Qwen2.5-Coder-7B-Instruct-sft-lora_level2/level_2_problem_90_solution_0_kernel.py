import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* weight, float* output, int N, int C_in, int D_in, int H_in, int W_in, int C_out, int D_out, int H_out, int W_out) {
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = blockIdx.y * blockDim.y + threadIdx.y;
    int d_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N && c_out < C_out && d_out < D_out && h_out < H_out && w_out < W_out) {
        float sum = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int d_in = 0; d_in < D_in; ++d_in) {
                for (int h_in = 0; h_in < H_in; ++h_in) {
                    for (int w_in = 0; w_in < W_in; ++w_in) {
                        sum += input[n * C_in * D_in * H_in * W_in + c_in * D_in * H_in * W_in + d_in * H_in * W_in + h_in * W_in + w_in] *
                               weight[c_out * C_in * D_in * H_in * W_in + c_in * D_in * H_in * W_in + d_in * H_in * W_in + h_in * W_in + w_in];
                    }
                }
            }
        }
        output[n * C_out * D_out * H_out * W_out + c_out * D_out * H_out * W_out + d_out * H_out * W_out + h_out * W_out + w_out] = sum;
    }
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight) {
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

    const int block_size = 16;
    const int num_blocks_d_out = (D_out + block_size - 1) / block_size;
    const int num_blocks_h_out = (H_out + block_size - 1) / block_size;
    const int num_blocks_w_out = (W_out + block_size - 1) / block_size;

    dim3 grid(num_blocks_d_out, num_blocks_h_out, num_blocks_w_out);
    dim3 block(block_size, block_size, block_size);

    conv3d_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out);

    return output;
}
"""

conv3d_cpp_source = (
    "torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 3D convolution
conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that uses custom CUDA kernels for 3D convolution and other operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = conv3d
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.clamp = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x, self.conv.weight)
        x = self.relu(x)
        x = x + self.sum_tensor
        x = self.clamp(x)
        x = self.gelu(x)
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 128
    in_channels = 8
    out_channels = 64
    depth, height, width = 16, 64, 64
    kernel_size = 3
    sum_tensor_shape = (out_channels, 1, 1, 1)

    model_new = ModelNew(in_channels, out_channels, kernel_size, sum_tensor_shape)
    inputs = torch.rand(batch_size, in_channels, depth, height, width)
    outputs = model_new(inputs)
    print(outputs.shape)