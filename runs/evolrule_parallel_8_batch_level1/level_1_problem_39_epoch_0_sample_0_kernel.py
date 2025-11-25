import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2_norm_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void l2_norm_kernel(const float* x, float* out, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* row = x + batch_idx * dim;
    float* out_row = out + batch_idx * dim;

    // Compute partial sum of squares
    float sum = 0.0f;
    for (int i = tid; i < dim; i += stride) {
        float val = row[i];
        sum += val * val;
    }

    // Use shared memory for partial sums
    extern __shared__ float sdata[];
    sdata[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The total norm squared is in sdata[0]
    float norm_sq = sdata[0];
    float norm = sqrtf(norm_sq);

    // Now, divide each element by the norm
    for (int i = tid; i < dim; i += stride) {
        out_row[i] = row[i] / norm;
    }
}

torch::Tensor l2_norm_cuda(torch::Tensor x) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);
    auto out = torch::empty_like(x);

    const int block_size = 1024;
    const dim3 grid(batch_size);
    const dim3 block(block_size);
    const size_t shared_mem = block_size * sizeof(float);

    l2_norm_kernel<<<grid, block, shared_mem, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);

    return out;
}
"""

l2_norm_cpp_source = (
    "torch::Tensor l2_norm_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code
l2_norm_extension = load_inline(
    name="l2_norm_cuda",
    cpp_sources=l2_norm_cpp_source,
    cuda_sources=l2_norm_source,
    functions=["l2_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_norm = l2_norm_extension

    def forward(self, x):
        return self.l2_norm.l2_norm_cuda(x)