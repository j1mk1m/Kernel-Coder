import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for L1 normalization fused into a single kernel
l1_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_l1_norm_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, const int batch_size, const int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ scalar_t shared_sum[256]; // Assuming max threads per block is 1024, but 256 is safer for SM sizes

    // Compute absolute values and accumulate sum per batch
    scalar_t sum = 0.0;
    for (int i = tid; i < dim; i += blockDim.x) {
        scalar_t val = x[batch_idx * dim + i];
        sum += fabs(val);
    }

    // Thread block reduction to compute sum for each batch
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sum += shared_sum[tid + s];
        }
    }
    __syncthreads();

    if (tid == 0) {
        shared_sum[0] = sum;
    }
    __syncthreads();

    // Compute mean and normalize
    scalar_t mean = shared_sum[0] / dim;
    if (mean == 0.0) { // Avoid division by zero, though input is from torch.rand()
        mean = 1.0;
    }

    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = batch_idx * dim + i;
        y[idx] = x[idx] / mean;
    }
}

torch::Tensor fused_l1_norm_cuda(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);

    auto y = torch::empty_like(x);
    const int block_size = 256; // Optimal block size for most GPUs
    const dim3 grid(batch_size);
    const dim3 block(block_size);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "fused_l1_norm_cuda", ([&] {
        fused_l1_norm_kernel<scalar_t><<<grid, block>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            batch_size,
            dim);
    }));

    return y;
}
"""

l1_norm_cpp_source = """
torch::Tensor fused_l1_norm_cuda(torch::Tensor x);
"""

# Compile the custom CUDA kernel
fused_l1_norm = load_inline(
    name="fused_l1_norm",
    cpp_sources=l1_norm_cpp_source,
    cuda_sources=l1_norm_source,
    functions=["fused_l1_norm_cuda"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=["--expt-extended-lambda"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_norm = fused_l1_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1_norm.fused_l1_norm_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return []