import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size, int stride, int padding, int output_padding) {
    int batch_id = blockIdx.x / (depth_out * height_out * width_out);
    int depth_id = (blockIdx.x % (depth_out * height_out * width_out)) / (height_out * width_out);
    int height_id = ((blockIdx.x % (depth_out * height_out * width_out)) % (height_out * width_out)) / width_out;
    int width_id = (blockIdx.x % (depth_out * height_out * width_out)) % width_out;

    int input_depth_start = depth_id * stride - padding;
    int input_height_start = height_id * stride - padding;
    int input_width_start = width_id * stride - padding;

    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_depth = input_depth_start + i;
                int input_height = input_height_start + j;
                int input_width = input_width_start + k;

                if (input_depth >= 0 && input_depth < depth_in && input_height >= 0 && input_height < height_in && input_width >= 0 && input_width < width_in) {
                    int input_index = batch_id * in_channels * depth_in * height_in * width_in + (input_depth * height_in * width_in + input_height * width_in + input_width) * in_channels + blockIdx.y;
                    int output_index = batch_id * out_channels * depth_out * height_out * width_out + (depth_id * height_out * width_out + height_id * width_out + width_id) * out_channels + (i * height_out * width_out + j * width_out + k) * out_channels + blockIdx.z;
                    atomicAdd(&output[output_index], input[input_index] * weight[i * kernel_size * kernel_size + j * kernel_size + k]);
                }
            }
        }
    }
}

torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);
    auto depth_out = (depth_in - 1) * stride - 2 * padding + output_padding[0] + kernel_size;
    auto height_out = (height_in - 1) * stride - 2 * padding + output_padding[1] + kernel_size;
    auto width_out = (width_in - 1) * stride - 2 * padding + output_padding[2] + kernel_size;

    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * depth_out * height_out * width_out + block_size - 1) / block_size;

    conv_transpose_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth_in, height_in, width_in, depth_out, height_out, width_out, kernel_size, stride, padding, output_padding);

    return output;
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding);"
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


# Define the custom CUDA kernel for GELU activation
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float gelu_device(float x) {
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu_device(input[idx]);
    }
}

void gelu_cuda(torch::Tensor input, torch::Tensor output) {
    auto size = input.numel();

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
}
"""

gelu_cpp_source = (
    "void gelu_cuda(torch::Tensor input, torch::Tensor output);"
)

# Compile the inline CUDA code for GELU activation
gelu = load_inline(
    name="gelu",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = gelu

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_cuda(x, self.weight, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        self.gelu.gelu_cuda(x, x)
        return x

# Example usage
batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size)
inputs = get_inputs()[0].cuda()
outputs = model_new(inputs)
print(outputs.shape)