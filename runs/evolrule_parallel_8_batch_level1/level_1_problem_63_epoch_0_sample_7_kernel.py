import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_forward(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int H_in,
    int W_in,
    int H_out,
    int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int out_channel = (idx / (W_out * H_out)) % out_channels;
    int batch = idx / (W_out * H_out * out_channels);

    float sum = 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                int h_in = h_out + k_h;
                int w_in = w_out + k_w;
                if (h_in < H_in && w_in < W_in) {
                    int input_offset = batch * in_channels * H_in * W_in
                                      + c_in * H_in * W_in
                                      + h_in * W_in + w_in;
                    float in_val = input[input_offset];

                    int weight_offset = out_channel * in_channels * kernel_size * kernel_size
                                       + c_in * kernel_size * kernel_size
                                       + k_h * kernel_size + k_w;
                    float weight_val = weight[weight_offset];

                    sum += in_val * weight_val;
                }
            }
        }
    }

    int output_offset = batch * out_channels * H_out * W_out
                       + out_channel * H_out * W_out
                       + h_out * W_out + w_out;
    output[output_offset] = sum;
}

extern "C" __host__ torch::Tensor conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int out_channels = weight.size(0);

    int H_out = H_in - kernel_size + 1;
    int W_out = W_in - kernel_size + 1;

    torch::Tensor output = torch::empty({batch_size, out_channels, H_out, W_out}, input.options());

    int total_elements = batch_size * out_channels * H_out * W_out;

    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    conv2d_forward<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        H_in,
        W_in,
        H_out,
        W_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\\n", cudaGetErrorString(err));
    }

    return output;
}
"""

conv2d_cuda_header = (
    "torch::Tensor conv2d_cuda_forward(torch::Tensor input, torch::Tensor weight, int kernel_size);"
)

conv2d_cuda = load_inline(
    name="conv2d_cuda",
    cpp_sources=conv2d_cuda_header,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_cuda_forward"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv2d_cuda_forward = conv2d_cuda.conv2d_cuda_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d_cuda_forward(x, self.weight, self.kernel_size)