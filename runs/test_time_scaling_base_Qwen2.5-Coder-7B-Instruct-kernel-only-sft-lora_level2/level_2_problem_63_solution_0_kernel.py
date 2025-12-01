import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Your custom CUDA kernels here

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        # Initialize the linear layer
        self.linear = nn.Linear(in_features, out_features)
        # Store the divisor as a module attribute
        self.divisor = divisor

    def forward(self, x):
        # Perform matrix multiplication using the linear layer
        x = self.linear(x)
        # Apply ReLU activation function using a custom CUDA kernel
        x = self.apply_relu_cuda(x)
        # Divide the result by the divisor using a custom CUDA kernel
        x = self.divide_by_divisor_cuda(x, self.divisor)
        return x
    
    # Define the custom CUDA kernel for ReLU activation
    relu_source = """
    // Include necessary headers
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    // Kernel definition for ReLU activation
    __global__ void relu_kernel(const float* input, float* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = max(input[idx], 0.0f);
        }
    }

    // Function to apply ReLU activation using CUDA
    torch::Tensor apply_relu_cuda(torch::Tensor input) {
        auto size = input.numel();
        auto output = torch::zeros_like(input);

        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;

        relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);

        return output;
    }
    """

    relu_cpp_source = (
        "torch::Tensor apply_relu_cuda(torch::Tensor input);"
    )

    # Compile the inline CUDA code for ReLU activation
    relu = load_inline(
        name="relu",
        cpp_sources=relu_cpp_source,
        cuda_sources=relu_source,
        functions=["apply_relu_cuda"],
        verbose=True,
        extra_cflags=[""],
        extra_ldflags=[""],
    )
    
    # Define the custom CUDA kernel for division by a constant
    divide_by_divisor_source = """
    // Include necessary headers
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    // Kernel definition for dividing by a constant
    __global__ void divide_by_divisor_kernel(const float* input, float* output, int size, float divisor) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = input[idx] / divisor;
        }
    }

    // Function to divide by a constant using CUDA
    torch::Tensor divide_by_divisor_cuda(torch::Tensor input, float divisor) {
        auto size = input.numel();
        auto output = torch::zeros_like(input);

        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;

        divide_by_divisor_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size, divisor);

        return output;
    }
    """

    divide_by_divisor_cpp_source = (
        "torch::Tensor divide_by_divisor_cuda(torch::Tensor input, float divisor);"
    )

    # Compile the inline CUDA code for division by a constant
    divide_by_divisor = load_inline(
        name="divide_by_divisor",
        cpp_sources=divide_by_divisor_cpp_source,
        cuda_sources=divide_by_divisor_source,
        functions=["divide_by_divisor_cuda"],
        verbose=True,
        extra_cflags=[""],
        extra_ldflags=[""],
    )