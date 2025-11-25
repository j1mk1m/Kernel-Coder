import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size) {
    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.z;
    int output_depth = blockIdx.x / (height * width);
    int output_height = (blockIdx.x % (height * width)) / width;
    int output_width = blockIdx.x % width;

    float sum = 0.0f;
    for (int d = 0; d < kernel_size; ++d) {
        for (int h = 0; h < kernel_size; ++h) {
            for (int w = 0; w < kernel_size; ++w) {
                int input_d = output_depth + d - kernel_size / 2;
                int input_h = output_height + h - kernel_size / 2;
                int input_w = output_width + w - kernel_size / 2;
                if (input_d >= 0 && input_d < depth && input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    int input_idx = batch_idx * in_channels * depth * height * width +
                                   channel_idx * depth * height * width +
                                   input_d * height * width +
                                   input_h * width +
                                   input_w;
                    int weight_idx = channel_idx * kernel_size * kernel_size * kernel_size +
                                    d * kernel_size * kernel_size +
                                    h * kernel_size +
                                    w;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = batch_idx * out_channels * depth * height * width +
                     channel_idx * depth * height * width +
                     output_depth * height * width +
                     output_height * width +
                     output_width;
    output[output_idx] = sum;
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth = input.size(2);
    auto height = input.size(3);
    auto width = input.size(4);
    auto kernel_size = weight.size(2);

    auto output = torch::zeros({batch_size, out_channels, depth, height, width}, input.options());

    dim3 threads_per_block(1, 1, 1);
    dim3 blocks_per_grid(depth * height * width, out_channels, in_channels);

    conv3d_kernel<<<blocks_per_grid, threads_per_block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_size);

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
    Optimized Model that uses a custom CUDA kernel for 3D convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = conv3d
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv(x, self.weight)
        x = self.norm(x)
        x = torch.min(x, torch.tensor(min_value, device=x.device))
        x = torch.clamp(x, min=min_value, max=max_value)
        x = self.dropout(x)
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 128
    in_channels = 3
    out_channels = 16
    depth, height, width = 16, 64, 64
    kernel_size = 3
    groups = 8
    min_value = 0.0
    max_value = 1.0
    dropout_p = 0.2

    model_new = ModelNew(in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p)
    inputs = get_inputs()
    outputs = model_new(inputs[0])
    print(outputs.shape)