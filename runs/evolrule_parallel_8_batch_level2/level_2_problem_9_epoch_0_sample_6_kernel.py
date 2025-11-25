import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for linear, subtraction, multiplication, and ReLU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

template <typename scalar_t>
__global__ void fused_linear_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    scalar_t subtract_value,
    scalar_t multiply_value) {

    int batch_id = blockIdx.x;
    int out_id = threadIdx.x;

    scalar_t sum = 0.0;
    for (int i = 0; i < in_features; ++i) {
        sum += input[batch_id * in_features + i] * weight[i * out_features + out_id];
    }

    // Apply bias (if any)
    // Note: The original model's linear layer includes bias by default
    sum += bias[out_id];

    // Apply subtraction and multiplication
    sum = (sum - subtract_value) * multiply_value;

    // Apply ReLU
    output[batch_id * out_features + out_id] = fmaxf(sum, 0.0);
}

torch::Tensor fused_linear_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value) {

    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(1);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int threads = out_features;
    const int blocks = batch_size;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_linear_relu_cuda", ([&] {
        fused_linear_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features,
            subtract_value,
            multiply_value);
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

cpp_source = """
torch::Tensor fused_linear_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float subtract_value,
    float multiply_value);
"""

# Compile the fused kernel
fused_linear_relu = load_inline(
    name="fused_linear_relu",
    cpp_sources=cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_linear_relu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value
        self.fused_kernel = fused_linear_relu

    def forward(self, x):
        # Extract parameters and constants
        weight = self.linear.weight.t().contiguous()  # Transpose for correct matrix multiplication
        bias = self.linear.bias.contiguous()
        subtract_value = self.subtract_value
        multiply_value = self.multiply_value

        # Run fused kernel
        return self.fused_kernel.fused_linear_relu_cuda(
            x, weight, bias, subtract_value, multiply_value
        )

# Keep the same get_inputs and get_init_inputs functions as original
def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]