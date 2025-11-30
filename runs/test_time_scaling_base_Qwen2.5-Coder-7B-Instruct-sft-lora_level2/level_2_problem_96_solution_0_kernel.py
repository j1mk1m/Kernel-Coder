import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed 3D convolution
transposed_convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_convolution_kernel(const float* input, float* output, int in_channels, int out_channels, int kernel_size, int stride, int padding, int batch_size, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out) {
    int b = blockIdx.x / (height_out * width_out);
    int h = (blockIdx.x % (height_out * width_out)) / width_out;
    int w = blockIdx.x % width_out;

    for (int c_out = 0; c_out < out_channels; ++c_out) {
        int k_h = blockIdx.y;
        int k_w = blockIdx.z;
        int c_in = threadIdx.x;

        if (c_in >= in_channels || k_h >= kernel_size || k_w >= kernel_size) continue;

        int i_d = h * stride - padding + k_h;
        int i_h = w * stride - padding + k_w;

        if (i_d >= 0 && i_d < depth_in && i_h >= 0 && i_h < height_in) {
            int i_w = blockIdx.w;
            int o_d = b * stride + i_d;
            int o_h = b * stride + i_h;
            int o_w = b * stride + i_w;

            atomicAdd(&output[o_d * out_channels * depth_out * height_out * width_out + o_h * out_channels * depth_out * height_out + o_w * out_channels * depth_out + c_out], input[b * in_channels * depth_in * height_in * width_in + c_in * depth_in * height_in * width_in + i_d * height_in * width_in + i_h * width_in + i_w]);
        }
    }
}

torch::Tensor transposed_convolution_cuda(torch::Tensor input, int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    auto batch_size = input.size(0);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);
    auto depth_out = (depth_in - 1) * stride + 1;
    auto height_out = (height_in - 1) * stride + 1;
    auto width_out = (width_in - 1) * stride + 1;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 32;
    const int grid_size = batch_size * height_out * width_out;

    dim3 threads(block_size);
    dim3 blocks(grid_size);

    transposed_convolution_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), in_channels, out_channels, kernel_size, stride, padding, batch_size, depth_in, height_in, width_in, depth_out, height_out, width_out);

    return output;
}
"""

transposed_convolution_cpp_source = (
    "torch::Tensor transposed_convolution_cuda(torch::Tensor input, int in_channels, int out_channels, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code for transposed 3D convolution
transposed_convolution = load_inline(
    name="transposed_convolution",
    cpp_sources=transposed_convolution_cpp_source,
    cuda_sources=transposed_convolution_source,
    functions=["transposed_convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.transposed_convolution = transposed_convolution

    def forward(self, x):
        x = self.transposed_convolution.transposed_convolution_cuda(x, in_channels, out_channels, kernel_size, stride, padding)
        x = x * self.scale
        x = F.max_pool3d(x, kernel_size=maxpool_kernel_size)
        x = F.avg_pool3d(x, kernel_size=(1, 1, 1))
        x = torch.clamp(x, min=0, max=1)
        return x

# Initialize the model and inputs
model_new = ModelNew(*get_init_inputs())
inputs = get_inputs()

# Forward pass through the model
output = model_new(inputs[0])
print(output.shape)