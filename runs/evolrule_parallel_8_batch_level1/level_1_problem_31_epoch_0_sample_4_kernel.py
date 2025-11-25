import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

elementwise_elu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CHECK(status) do { \\
    cudaError_t err = status; \\
    if (err != cudaSuccess) { \\
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \\
        throw std::runtime_error("CUDA error"); \\
    } \\
} while (0)

template <typename scalar_t>
__global__ void elu_kernel(const scalar_t* x_data, scalar_t* y_data, const scalar_t alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        scalar_t x_val = x_data[idx];
        y_data[idx] = (x_val > 0) ? x_val : alpha * (exp(x_val) - static_cast<scalar_t>(1));
    }
}

torch::Tensor elementwise_elu_cuda(torch::Tensor x, float alpha) {
    auto n = x.numel();
    auto y = torch::empty_strided(x.sizes(), x.strides(), x.options());

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "elu_cuda", ([&] {
        scalar_t alpha_val = static_cast<scalar_t>(alpha);
        elu_kernel<scalar_t><<<num_blocks, block_size>>>(x.data<scalar_t>(), y.data<scalar_t>(), alpha_val, n);
    }));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return y;
}
"""

elementwise_elu_cpp_source = (
    "#include <torch/extension.h>\n"
    "torch::Tensor elementwise_elu_cuda(torch::Tensor x, float alpha);"
)

elementwise_elu = load_inline(
    name="elementwise_elu",
    cpp_sources=elementwise_elu_cpp_source,
    cuda_sources=elementwise_elu_source,
    functions=["elementwise_elu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.elementwise_elu = elementwise_elu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elementwise_elu.elementwise_elu_cuda(x, self.alpha)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()  # Ensure input is on CUDA
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization