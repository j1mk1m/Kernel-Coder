import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scalar multiplication
scalar_mult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void scalar_mult_kernel(const T* a, T scalar, T* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * scalar;
    }
}

template <typename T>
torch::Tensor scalar_mult_cuda(torch::Tensor a, T scalar) {
    auto size = a.numel();
    auto out = torch::empty_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    scalar_mult_kernel<T><<<num_blocks, block_size>>>(a.data_ptr<T>(), scalar, out.data_ptr<T>(), size);

    return out;
}

torch::Tensor scalar_mult(torch::Tensor a, ScalarType scalar) {
    auto dtype = a.scalar_type();
    if (dtype == torch::kFloat32) {
        return scalar_mult_cuda<float>(a, scalar.to<float>());
    } else if (dtype == torch::kFloat64) {
        return scalar_mult_cuda<double>(a, scalar.to<double>());
    } else {
        TORCH_CHECK(false, "Unsupported tensor type");
    }
}
"""

scalar_mult_cpp_source = """
#include <torch/extension.h>
torch::Tensor scalar_mult(torch::Tensor a, at::Scalar scalar);
"""

# Compile the inline CUDA code for scalar multiplication
scalar_mult = load_inline(
    name="scalar_mult",
    cpp_sources=scalar_mult_cpp_source,
    cuda_sources=scalar_mult_source,
    functions=["scalar_mult"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.scalar_mult = scalar_mult

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        # Convert scalar to appropriate type based on tensor's dtype
        scalar = A.dtype.type(s)
        return self.scalar_mult(A, scalar)

def get_inputs():
    # Use the original function for generating inputs
    A = torch.rand(M, N).cuda()
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []