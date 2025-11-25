import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cumprod_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void cumprod_kernel(const scalar_t* __restrict__ input, 
                              scalar_t* __restrict__ output,
                              int N) {
    const int row = blockIdx.x;
    const int start_idx = row * N;
    scalar_t product = static_cast<scalar_t>(1.0);
    for (int i = 0; i < N; ++i) {
        const int idx = start_idx + i;
        product *= input[idx];
        output[idx] = product;
    }
}

torch::Tensor cumprod_cuda(torch::Tensor input, int dim) {
    if (dim != 1) {
        AT_ERROR("Only dim=1 is supported");
    }
    int batch_size = input.size(0);
    int N = input.size(1);

    auto output = torch::empty_like(input);

    dim3 blocks(batch_size);
    dim3 threads(1);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N
        );
    }));

    return output;
}
"""

cumprod_header = """
torch::Tensor cumprod_cuda(torch::Tensor input, int dim);
"""

cumprod = load_inline(
    name="cumprod",
    cpp_sources=cumprod_header,
    cuda_sources=cumprod_source,
    functions=["cumprod_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.cumprod = cumprod  # The compiled CUDA module

    def forward(self, x):
        return self.cumprod.cumprod_cuda(x, self.dim)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [1]  # dim=1 as per the original setup