import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for subtraction, squaring, and mean reduction
subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void subtract_kernel(const float* predictions, const float* targets, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = predictions[idx] - targets[idx];
    }
}

torch::Tensor subtract_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto size = predictions.numel();
    auto result = torch::zeros_like(predictions);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    subtract_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), result.data_ptr<float>(), size);

    return result;
}
"""

square_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void square_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
}

torch::Tensor square_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    square_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output;
}
"""

mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

torch::Tensor mean_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros({1}, input.options());

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    mean_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

    return output / static_cast<float>(size);
}
"""

subtract_cpp_source = (
    "torch::Tensor subtract_cuda(torch::Tensor predictions, torch::Tensor targets);"
)
square_cpp_source = (
    "torch::Tensor square_cuda(torch::Tensor input);"
)
mean_cpp_source = (
    "torch::Tensor mean_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for subtraction, squaring, and mean reduction
subtract = load_inline(
    name="subtract",
    cpp_sources=subtract_cpp_source,
    cuda_sources=subtract_source,
    functions=["subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

square = load_inline(
    name="square",
    cpp_sources=square_cpp_source,
    cuda_sources=square_source,
    functions=["square_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

mean = load_inline(
    name="mean",
    cpp_sources=mean_cpp_source,
    cuda_sources=mean_source,
    functions=["mean_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.subtract = subtract
        self.square = square
        self.mean = mean

    def forward(self, predictions, targets):
        diff = self.subtract.subtract_cuda(predictions, targets)
        squared_diff = self.square.square_cuda(diff)
        mse = self.mean.mean_cuda(squared_diff)
        return mse


def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape) * scale, torch.rand(batch_size, *input_shape)]


def get_init_inputs():
    return []