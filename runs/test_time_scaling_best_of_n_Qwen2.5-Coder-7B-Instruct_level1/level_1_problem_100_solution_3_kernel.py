from torch.utils.cpp_extension import load_inline

example_source = """
#include <torch/extension.h>

// Define a simple CUDA kernel
__global__ void example_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= 2;
    }
}

// Wrap the kernel in a Python function
torch::Tensor example_cuda(torch::Tensor x) {
    int n = x.size(0);
    example_kernel<<<n / 256 + 1, 256>>>(x.data_ptr<float>(), n);
    return x;
}
"""

example_cpp_source = "torch::Tensor example_cuda(torch::Tensor x);"

# Compile and link the CUDA code
example = load_inline(name="example", cpp_sources=example_cpp_source, cuda_sources=example_source, functions=["example_cuda"], verbose=True)