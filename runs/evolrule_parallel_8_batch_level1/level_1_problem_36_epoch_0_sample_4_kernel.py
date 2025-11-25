import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for RMSNorm
rmsnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void rmsnorm_forward_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int batch_size,
    const int features,
    const int dim1,
    const int dim2,
    const float eps,
    const int total_features) {

    extern __shared__ scalar_t shared_data[];

    const int batch_idx = blockIdx.x;
    const int dim1_idx = blockIdx.y;
    const int dim2_idx = blockIdx.z;

    const int base_offset = batch_idx * features * dim1 * dim2 +
                           dim1_idx * dim2 + dim2_idx;

    // Load data into shared memory
    scalar_t sum = 0.0;
    for (int f = threadIdx.x; f < features; f += blockDim.x) {
        const int offset = base_offset + f * dim1 * dim2;
        const scalar_t val = x[offset];
        sum += val * val;
    }

    __syncthreads();

    // Block reduction using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        shared_data[0] = static_cast<scalar_t>(sqrt(sum / total_features + eps));
    }
    __syncthreads();

    const scalar_t inv_rms = 1.0 / shared_data[0];

    // Write output
    for (int f = threadIdx.x; f < features; f += blockDim.x) {
        const int offset = base_offset + f * dim1 * dim2;
        y[offset] = x[offset] * inv_rms;
    }
}

torch::Tensor rmsnorm_forward_cuda(torch::Tensor x, float eps) {
    const int batch_size = x.size(0);
    const int features = x.size(1);
    const int dim1 = x.size(2);
    const int dim2 = x.size(3);
    const int total_features = features;

    auto y = torch::empty_like(x);

    const dim3 blocks(batch_size, dim1, dim2);
    const dim3 threads(std::min(features, 256));
    const size_t shared_size = sizeof(float) * (threads.x + 1);

    rmsnorm_forward_kernel<float><<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size,
        features,
        dim1,
        dim2,
        eps,
        total_features
    );

    return y;
}
"""

rmsnorm_cpp_source = """
torch::Tensor rmsnorm_forward_cuda(torch::Tensor x, float eps);
"""

# Compile the inline CUDA code
rmsnorm = load_inline(
    name="rmsnorm",
    cpp_sources=rmsnorm_cpp_source,
    cuda_sources=rmsnorm_source,
    functions=["rmsnorm_forward_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.rmsnorm_forward = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move data to CUDA and ensure contiguous for kernel access
        x_cuda = x.contiguous().cuda()
        return self.rmsnorm_forward.rmsnorm_forward_cuda(x_cuda, self.eps).cpu()