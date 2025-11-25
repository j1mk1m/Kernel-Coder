import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for convolution
convolution_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void convolution_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = blockIdx.x / width;
    int w_out = blockIdx.x % width;

    int c_in = threadIdx.z;
    int k_h = threadIdx.y;
    int k_w = threadIdx.x;

    float sum = 0.0f;
    if (c_in < in_channels && k_h < kernel_size && k_w < kernel_size) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int input_idx = n * in_channels * height * width + c_in * height * width + i * width + j;
                int weight_idx = c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + k_h * kernel_size + k_w;
                int output_idx = n * out_channels * height * width + c_out * height * width + h_out * width + w_out;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        output[blockIdx.x] = sum;
    }
}

torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size) {
    auto output = torch::zeros({batch_size, out_channels, height, width}, input.options());

    dim3 grid((height * width + 255) / 256, out_channels, batch_size);
    dim3 block(kernel_size * kernel_size, 1, in_channels);

    convolution_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, height, width, kernel_size);

    return output;
}
"""

convolution_cpp_source = (
    "torch::Tensor convolution_cuda(torch::Tensor input, torch::Tensor weight, int batch_size, int in_channels, int out_channels, int height, int width, int kernel_size);"
)

# Compile the inline CUDA code for convolution
convolution = load_inline(
    name="convolution",
    cpp_sources=convolution_cpp_source,
    cuda_sources=convolution_source,
    functions=["convolution_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for HardSwish
hardswish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardswish_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * (input[idx] > 0 ? ((input[idx] < 6 ? input[idx] : 6)) / 6 : 0);
    }
}

torch::Tensor hardswish_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (input.numel() + block_size - 1) / block_size;

    hardswish_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());

    return output;
}
"""

hardswish_cpp_source = (
    "torch::Tensor hardswish_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for HardSwish
hardswish = load_inline(
    name="hardswish",
    cpp_sources=hardswish_cpp_source,
    cuda_sources=hardswish_source,
    functions=["hardswish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = convolution
        self.hardswish = hardswish

    def forward(self, x):
        x = self.conv.convolution_cuda(x, self.weight, batch_size, in_channels, out_channels, height, width, kernel_size)
        x = self.hardswish.hardswish_cuda(x)
        x = torch.relu(x)
        return x

# Example usage
if __name__ == "__main__":
    model = ModelNew(in_channels, out_channels, kernel_size)
    inputs = get_inputs()
    outputs = model(inputs[0])
    print(outputs.shape)