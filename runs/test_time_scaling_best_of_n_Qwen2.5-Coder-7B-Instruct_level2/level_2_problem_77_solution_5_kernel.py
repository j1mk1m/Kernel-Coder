import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void conv_transpose_kernel(const float* input, const float* weight, float* output, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out) {
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    int ic = blockIdx.z * blockDim.z + threadIdx.z;
    if (oc >= out_channels || ic >= in_channels) return;

    int o_d = blockIdx.x / (height_out * width_out);
    int o_h = (blockIdx.x % (height_out * width_out)) / width_out;
    int o_w = blockIdx.x % width_out;

    int i_d_start = max(o_d * depth_out - depth_in + 1, 0);
    int i_d_end = min(o_d * depth_out + 1, depth_in);
    int i_h_start = max(o_h * height_out - height_in + 1, 0);
    int i_h_end = min(o_h * height_out + 1, height_in);
    int i_w_start = max(o_w * width_out - width_in + 1, 0);
    int i_w_end = min(o_w * width_out + 1, width_in);

    float sum = 0.0f;
    for (int d = i_d_start; d < i_d_end; ++d) {
        for (int h = i_h_start; h < i_h_end; ++h) {
            for (int w = i_w_start; w < i_w_end; ++w) {
                int i_idx = ((o_d * height_out + o_h) * width_out + o_w) * in_channels * depth_in * height_in * width_in +
                           d * height_in * width_in +
                           h * width_in +
                           w;
                int w_idx = ic * depth_in * height_in * width_in +
                           d * height_in * width_in +
                           h * width_in +
                           w;
                sum += input[i_idx] * weight[w_idx];
            }
        }
    }

    output[o_d * height_out * width_out + o_h * width_out + o_w] = sum;
}

torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight, int depth_out, int height_out, int width_out) {
    auto out_channels = weight.size(0);
    auto in_channels = weight.size(1);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);

    auto output = torch::zeros({depth_out, height_out, width_out}, input.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(divup(depth_out, threads.x), divup(height_out, threads.y), divup(in_channels, threads.z));

    conv_transpose_kernel<<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), in_channels, out_channels, depth_in, height_in, width_in, depth_out, height_out, width_out);

    return output;
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight, int depth_out, int height_out, int width_out);"
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
    """
    Optimized Model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_cuda(x, self.weight, self.depth_out, self.height_out, self.width_out)
        x = x * self.scale_factor
        x = self.batch_norm(x)
        x = self.global_avg_pool(x)
        return x