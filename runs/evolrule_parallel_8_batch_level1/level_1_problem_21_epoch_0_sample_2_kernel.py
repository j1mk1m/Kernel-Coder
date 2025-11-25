import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Sigmoid activation
sigmoid_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                              scalar_t* __restrict__ output,
                              int64_t size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        scalar_t x = input[tid];
        // Compute sigmoid using 1 / (1 + exp(-x))
        // Handle overflow/underflow cases for numerical stability
        if (x >= 20) {
            output[tid] = 1.0;
        } else if (x <= -20) {
            output[tid] = 0.0;
        } else {
            output[tid] = static_cast<scalar_t>(1.0 / (1.0 + exp(-x)));
        }
    }
}

std::tuple<torch::Tensor> sigmoid_forward(torch::Tensor input) {
    const auto size = input.numel();
    auto output = torch::empty_like(input);
    
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaGetLastError());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_forward", ([&] {
        using scalar_t = scalar_type;
        sigmoid_kernel<scalar_t><<<grid_size, block_size, 0, stream>>>(
            input.data<scalar_t>(), output.data<scalar_t>(), size);
    }));

    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(output);
}

TORCH_LIBRARY(my_ops, m) {
    m.def("sigmoid_forward", TORCH_FN(sigmoid_forward));
}
"""

# Define the CUDA source header
sigmoid_cpp_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
"""

# Compile the CUDA kernel
sigmoid_module = load_inline(
    name="sigmoid_cuda",
    cpp_sources=[sigmoid_cpp_source],
    cuda_sources=[sigmoid_source],
    extra_cuda_cflags=['-arch=sm_70'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Register the custom op with TorchScript
        x = x.cuda() if x.device != torch.device("cuda") else x
        return sigmoid_module.sigmoid_forward(x)[0]

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []