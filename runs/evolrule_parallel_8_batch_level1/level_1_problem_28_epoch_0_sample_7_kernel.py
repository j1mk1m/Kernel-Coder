import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for HardSigmoid
hardsigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const double* x, double* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double temp = x[idx] + 3.0;
        temp = max(0.0, temp);
        temp = min(temp, 6.0);
        y[idx] = temp / 6.0;
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor x) {
    auto options = torch::TensorOptions().dtype(torch::kDouble).device(x.device());
    auto y = torch::empty_like(x, options);
    int n = x.numel();
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    hardsigmoid_kernel<<<num_blocks, block_size>>>(x.data_ptr<double>(), y.data_ptr<double>(), n);
    cudaDeviceSynchronize();
    return y;
}
"""

hardsigmoid_cpp_source = (
    "torch::Tensor hardsigmoid_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code
hardsigmoid = load_inline(
    name="hardsigmoid",
    cpp_sources=hardsigmoid_cpp_source,
    cuda_sources=hardsigmoid_source,
    functions=["hardsigmoid_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardsigmoid_cuda = hardsigmoid  # access the module containing the kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to double and move to CUDA
        x_double = x.double()
        x_gpu = x_double.cuda()
        # Apply the custom CUDA kernel
        y_gpu = self.hardsigmoid_cuda.hardsigmoid_cuda(x_gpu)
        # Convert back to float and return to CPU
        return y_gpu.cpu().float()

# Ensure the same inputs as the original get_inputs()
# Note: The original get_inputs() returns CPU tensors of float32
# The above forward() correctly handles type and device conversions to match this