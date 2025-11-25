import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import numpy as np

# Custom CUDA kernel implementation for LeakyReLU
leaky_relu_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/macros/Macros.h>
#include <ATen/native/TensorIterator.h>

template <typename scalar_t>
__global__ void leaky_relu_kernel(const at::PackedTensorAccessor<scalar_t, 1> input,
                                 at::PackedTensorAccessor<scalar_t, 1> output,
                                 const scalar_t negative_slope, int64_t n) {
    CUDA_KERNEL_LOOP(i, n) {
        scalar_t val = input[i];
        output[i] = val > 0 ? val : val * negative_slope;
    }
}

std::tuple<at::Tensor> leaky_relu_cuda(const at::Tensor& input, const at::Scalar& negative_slope) {
    auto output = at::empty_like(input);
    auto input_acc = input.packed_accessor<>();
    auto output_acc = output.packed_accessor<>();

    int64_t n = input.numel();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "leaky_relu_cuda", [&] {
        using scalar_t = scalar_type;
        leaky_relu_kernel<scalar_t><<<grid_size, block_size>>>(
            input.packed_accessor<scalar_t, 1>(),
            output.packed_accessor<scalar_t, 1>(),
            negative_slope.to<scalar_t>(),
            n);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    });

    return std::make_tuple(output);
}
"""

# Header for C++ function
leaky_relu_cpp_source = (
    "#include <torch/extension.h>"
    "std::tuple<torch::Tensor> leaky_relu_cuda(const torch::Tensor& input, const at::Scalar& negative_slope);"
)

# Load the CUDA extension inline
leaky_relu = load_inline(
    name="leaky_relu",
    cpp_sources=[leaky_relu_cpp_source],
    cuda_sources=[leaky_relu_source],
    functions="leaky_relu_cuda",
    verbose=True,
    extra_cflags=["-DWITH_CUDA", "-xcuda"],
    extra_cuda_cflags=["-gencode=arch=compute_70,code=sm_70", "--use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
        self.leaky_relu_cuda = leaky_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to CUDA tensors if not already on GPU
        if x.is_cuda:
            input_gpu = x
        else:
            input_gpu = x.cuda()

        # Execute custom CUDA kernel
        output_tuple = self.leaky_relu_cuda.leaky_relu_cuda(input_gpu, self.negative_slope)
        output = output_tuple[0]

        # Handle CPU outputs if needed (though kernel requires CUDA tensors)
        if not x.is_cuda:
            return output.cpu()
        return output

# Ensure inputs are on correct device
def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []