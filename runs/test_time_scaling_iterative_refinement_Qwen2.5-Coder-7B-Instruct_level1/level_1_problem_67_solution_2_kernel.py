import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 1D convolution
convolution_1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void convolution_1d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int length, int out_channels, int kernel_size) {
    int b = blockIdx.x;
    int o = blockIdx.y;
    int i = blockIdx.z;

    __shared__ float s_input[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_weight[TILE_WIDTH][TILE_WIDTH];

    float sum = 0.0f;

    for (int k = 0; k < kernel_size; k++) {
        int ii = i + k;
        if (ii >= length) continue;

        int gi = b * in_channels * length + i * length + ii;
        s_input[threadIdx.y][threadIdx.x] = input[gi];

        int gw = o * out_channels * kernel_size + o * kernel_size + k;
        s_weight[threadIdx.y][threadIdx.x] = weight[gw];

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++) {
            sum += s_input[j][threadIdx.x] * s_weight[threadIdx.y][j];
        }

        __syncthreads();
    }

    int go = b * out_channels * length + o * length + i;
    output[go] = sum;
}

torch::Tensor convolution_1d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto length = input.size(2);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(1);

    auto output = torch::zeros({batch_size, out_channels, length}, input.options());

    dim3 grid_dim(batch_size, out_channels, length);
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);

    convolution_1d_kernel<<<grid_dim, block_dim>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, length, out_channels, kernel_size);

    return output;
}
"""

convolution_1d_cpp_source = (
    "torch::Tensor convolution_1d_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for 1D convolution
convolution_1d = load_inline(
    name="convolution_1d",
    cpp_sources=convolution_1d_cpp_source,
    cuda_sources=convolution_1d_source,
    functions=["convolution_1d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.convolution_1d = convolution_1d

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).
            weight (torch.Tensor): Weight tensor of shape (out_channels, kernel_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.convolution_1d.convolution_1d_cuda(x, weight)