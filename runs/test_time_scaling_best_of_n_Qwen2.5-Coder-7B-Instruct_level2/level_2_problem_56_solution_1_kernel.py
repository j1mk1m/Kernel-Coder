import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for linear transformation
linear_transform_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_transform_kernel(const float* input, const float* weight, float* output, int batch_size, int input_size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_size) {
        int row = idx / hidden_size;
        int col = idx % hidden_size;
        output[idx] = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            output[idx] += input[row * input_size + i] * weight[i * hidden_size + col];
        }
    }
}

torch::Tensor linear_transform_cuda(torch::Tensor input, torch::Tensor weight) {
    auto batch_size = input.size(0);
    auto input_size = input.size(1);
    auto hidden_size = weight.size(1);
    auto output = torch::zeros({batch_size, hidden_size}, input.options());

    const int block_size = 256;
    const int num_blocks = (batch_size * hidden_size + block_size - 1) / block_size;

    linear_transform_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), batch_size, input_size, hidden_size);

    return output;
}
"""

linear_transform_cpp_source = (
    "torch::Tensor linear_transform_cuda(torch::Tensor input, torch::Tensor weight);"
)

# Compile the inline CUDA code for linear transformation
linear_transform = load_inline(
    name="linear_transform",
    cpp_sources=linear_transform_cpp_source,
    cuda_sources=linear_transform_source,
    functions=["linear_transform_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for sigmoid activation
sigmoid_activation_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sigmoid_activation_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + exp(-input[idx]));
    }
}

torch::Tensor sigmoid_activation_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    sigmoid_activation_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

sigmoid_activation_cpp_source = (
    "torch::Tensor sigmoid_activation_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for sigmoid activation
sigmoid_activation = load_inline(
    name="sigmoid_activation",
    cpp_sources=sigmoid_activation_cpp_source,
    cuda_sources=sigmoid_activation_source,
    functions=["sigmoid_activation_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))

    def forward(self, x):
        x = linear_transform.linear_transform_cuda(x, self.weight)
        x = sigmoid_activation.sigmoid_activation_cuda(x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size]