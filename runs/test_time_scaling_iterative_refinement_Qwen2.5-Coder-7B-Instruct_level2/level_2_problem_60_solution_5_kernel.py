import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D transposed convolution
conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth_in, int height_in, int width_in, int depth_out, int height_out, int width_out, int kernel_size, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * depth_out * height_out * width_out) return;

    int b = idx / (out_channels * depth_out * height_out * width_out);
    int c = (idx / (depth_out * height_out * width_out)) % out_channels;
    int d_out = (idx / (height_out * width_out)) % depth_out;
    int h_out = (idx / width_out) % height_out;
    int w_out = idx % width_out;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int d_in = d_out * stride - padding + k;
                int h_in = h_out * stride - padding + i;
                int w_in = w_out * stride - padding + j;
                if (d_in >= 0 && d_in < depth_in && h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                    int input_idx = b * in_channels * depth_in * height_in * width_in + c * depth_in * height_in * width_in + d_in * height_in * width_in + h_in * width_in + w_in;
                    int weight_idx = c * in_channels * kernel_size * kernel_size * kernel_size + (c * kernel_size * kernel_size + k * kernel_size + i) * kernel_size + j;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    output[idx] = sum;
}

torch::Tensor conv_transpose_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto depth_in = input.size(2);
    auto height_in = input.size(3);
    auto width_in = input.size(4);
    auto depth_out = (depth_in - 1) * stride + kernel_size - 2 * padding;
    auto height_out = (height_in - 1) * stride + kernel_size - 2 * padding;
    auto width_out = (width_in - 1) * stride + kernel_size - 2 * padding;
    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * depth_out * height_out * width_out + block_size - 1) / block_size;

    conv_transpose_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth_in, height_in, width_in, depth_out, height_out, width_out, kernel_size, stride, padding);

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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = conv_transpose
        self.swish = lambda x: x * torch.sigmoid(x)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.hardswish = lambda x: F.relu6(x + 3) / 6

    def forward(self, x):
        x = self.conv_transpose.conv_transpose_cuda(x, self.weight)
        x = self.swish(x)
        x = self.group_norm(x)
        x = self.hardswish(x)
        return x

# Initialize weights for the convolutional layer
def init_weights(m):
    if isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model = Model(in_channels, out_channels, kernel_size, stride, padding, groups, eps)
model.apply(init_weights)

model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, groups, eps)
model_new.apply(init_weights)

# Get inputs
inputs = get_inputs()

# Forward pass through original model
output_ref = model(*inputs)

# Forward pass through new model
output_new = model_new(*inputs)

# Check if outputs are equal
assert torch.allclose(output_ref, output_new, atol=1e-5)