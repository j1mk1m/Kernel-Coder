import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D convolution
conv3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* weight, float* output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size) {
    int batch_id = blockIdx.z * blockDim.z + threadIdx.z;
    int out_channel_id = blockIdx.y * blockDim.y + threadIdx.y;
    int in_channel_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_id >= batch_size || out_channel_id >= out_channels || in_channel_id >= in_channels) {
        return;
    }

    int pad_depth = kernel_size / 2;
    int pad_height = kernel_size / 2;
    int pad_width = kernel_size / 2;

    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int input_idx = (batch_id * in_channels + in_channel_id) * depth * height * width +
                                (d + pad_depth) * height * width + (h + pad_height) * width + (w + pad_width);
                float sum = 0.0f;
                for (int kd = 0; kd < kernel_size; ++kd) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int weight_idx = (out_channel_id * in_channels + in_channel_id) * kernel_size * kernel_size * kernel_size +
                                            kd * kernel_size * kernel_size + kh * kernel_size + kw;
                            int weight_value = weight[weight_idx];
                            int input_d = d - pad_depth + kd;
                            int input_h = h - pad_height + kh;
                            int input_w = w - pad_width + kw;
                            if (input_d >= 0 && input_d < depth && input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                sum += input[input_idx] * weight_value;
                            }
                        }
                    }
                }
                int output_idx = (batch_id * out_channels + out_channel_id) * depth * height * width +
                                 d * height * width + h * width + w;
                atomicAdd(&output[output_idx], sum);
            }
        }
    }
}

void conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size) {
    const int block_size = 32;
    const int num_blocks_out_channel = (out_channels + block_size - 1) / block_size;
    const int num_blocks_in_channel = (in_channels + block_size - 1) / block_size;
    const int num_blocks_batch = (batch_size + block_size - 1) / block_size;

    dim3 grid(num_blocks_in_channel, num_blocks_out_channel, num_blocks_batch);
    dim3 block(block_size, block_size, 1);

    conv3d_kernel<<<grid, block>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, in_channels, out_channels, depth, height, width, kernel_size);
}
"""

conv3d_cpp_source = (
    "void conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor output, int batch_size, int in_channels, int out_channels, int depth, int height, int width, int kernel_size);"
)

# Compile the inline CUDA code for 3D convolution
conv3d = load_inline(
    name="conv3d",
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_source,
    functions=["conv3d_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for LeakyReLU
leakyrelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leakyrelu_kernel(const float* input, float* output, int size, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : negative_slope * input[idx];
    }
}

torch::Tensor leakyrelu_cuda(torch::Tensor input, float negative_slope) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    leakyrelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, negative_slope);

    return output;
}
"""

leakyrelu_cpp_source = (
    "torch::Tensor leakyrelu_cuda(torch::Tensor input, float negative_slope);"
)

# Compile the inline CUDA code for LeakyReLU
leakyrelu = load_inline(
    name="leakyrelu",
    cpp_sources=leakyrelu_cpp_source,
    cuda_sources=leakyrelu_source,
    functions=["leakyrelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Sum
sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(output, input[idx]);
    }
}

torch::Tensor sum_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros({1}, input.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sum_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

sum_cpp_source = (
    "torch::Tensor sum_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Sum
sum_op = load_inline(
    name="sum_op",
    cpp_sources=sum_cpp_source,
    cuda_sources=sum_source,
    functions=["sum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Clamp
clamp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void clamp_kernel(const float* input, float* output, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] < min_val ? min_val : (input[idx] > max_val ? max_val : input[idx]);
    }
}

torch::Tensor clamp_cuda(torch::Tensor input, float min_val, float max_val) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    clamp_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, min_val, max_val);

    return output;
}
"""

clamp_cpp_source = (
    "torch::Tensor clamp_cuda(torch::Tensor input, float min_val, float max_val);"
)

# Compile the inline CUDA code for Clamp
clamp_op = load_inline(
    name="clamp_op",
    cpp_sources=clamp_cpp_source,
    cuda_sources=clamp_source,
    functions=["clamp_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for GELU
gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ float gelu_device(float x) {
    return 0.5f * (x + tanhf(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = gelu_device(input[idx]);
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

gelu_cpp_source = (
    "torch::Tensor gelu_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for GELU
gelu_op = load_inline(
    name="gelu_op",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.register_buffer('conv_weight', self.conv.weight.detach())
        self.register_buffer('conv_bias', self.conv.bias.detach())

    def forward(self, x):
        # Perform 3D convolution using the custom CUDA kernel
        batch_size = x.size(0)
        in_channels = x.size(1)
        out_channels = self.conv.out_channels
        depth = x.size(2)
        height = x.size(3)
        width = x.size(4)
        kernel_size = self.conv.kernel_size[0]

        output = torch.zeros((batch_size, out_channels, depth, height, width), device=x.device)
        conv3d_forward_cuda(x, self.conv_weight, output, batch_size, in_channels, out_channels, depth, height, width, kernel_size)

        # Apply LeakyReLU using the custom CUDA kernel
        output = leakyrelu_cuda(output, negative_slope=0.2)

        # Sum with the pre-defined sum tensor using the custom CUDA kernel
        sum_result = sum_cuda(self.sum_tensor.view(-1)).item()
        output += sum_result

        # Clamp using the custom CUDA kernel
        output = clamp_cuda(output, min_val=-1.0, max_val=1.0)

        # Apply GELU using the custom CUDA kernel
        output = gelu_cuda(output)

        return output

# Example usage
model_new = ModelNew(in_channels, out_channels, kernel_size, sum_tensor_shape)
inputs = get_inputs()
outputs = model_new(inputs[0])
print(outputs.shape)