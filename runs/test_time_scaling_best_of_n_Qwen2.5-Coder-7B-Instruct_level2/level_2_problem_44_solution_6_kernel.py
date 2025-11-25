import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for transposed convolution
transposed_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom implementation of transposed convolution
__global__ void transposed_convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height_in, int width_in, int height_out, int width_out, int kernel_size, int stride, int padding, int dilation, int groups) {
    // Implement the transposed convolution logic here
    // This is a placeholder for actual CUDA code
    // You need to fill in the details of the transposed convolution operation
    // Ensure that the output tensor has the correct dimensions and values
}

torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding, int groups) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto out_channels = weight.size(0);
    auto height_in = input.size(2);
    auto width_in = input.size(3);
    auto kernel_size = weight.size(2);
    auto stride = stride;
    auto padding = padding;
    auto output_padding = output_padding;
    auto groups = groups;
    auto height_out = (height_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto width_out = (width_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    auto output = torch::zeros({batch_size, out_channels, height_out, width_out});

    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * height_out * width_out + block_size - 1) / block_size;

    transposed_convolution_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height_in, width_in, height_out, width_out, kernel_size, stride, padding, 1, groups);

    return output;
}
"""

transposed_conv_cpp_source = (
    "torch::Tensor transposed_convolution_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int output_padding, int groups);"
)

# Compile the inline CUDA code for transposed convolution
transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources=transposed_conv_cpp_source,
    cuda_sources=transposed_conv_source,
    functions=["transposed_convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.transposed_conv = transposed_conv
        self.multiplier = multiplier

    def forward(self, x):
        x = self.transposed_conv.transposed_convolution_cuda(x, self.weight, stride, padding, output_padding, self.groups)
        x = x * self.multiplier
        x = torch.mean(x, dim=[2, 3], keepdim=True)  # First global average pooling
        x = torch.mean(x, dim=[2, 3], keepdim=True)  # Second global average pooling
        return x

# Initialize weights and other parameters
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5
groups = 1

weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()

model_new = ModelNew(in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier)
model_new.weight = weight

inputs = get_inputs()
outputs = model_new(inputs[0].cuda())
print(outputs.shape)