import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

l2_normalize_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_norms_kernel(const float* x, float* norms, int batch_size, int dim) {
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    float sum = 0.0f;

    const float* x_row = x + row * dim;
    for (int i = tid; i < dim; i += stride) {
        float val = x_row[i];
        sum += val * val;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float norm = sqrt(sdata[0]);
        norms[row] = norm;
    }
}

__global__ void normalize_kernel(const float* x, const float* norms, float* out, int batch_size, int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim) return;

    int row = idx / dim;
    int col = idx % dim;

    float val = x[row * dim + col];
    float norm = norms[row];

    out[idx] = val / norm;
}

torch::Tensor l2_normalize_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);

    auto norms = torch::empty({batch_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto out = torch::empty_like(x);

    const int block_size = 256;
    dim3 compute_norms_blocks(batch_size);
    dim3 compute_norms_threads(block_size);
    int shared_size = block_size * sizeof(float);
    compute_norms_kernel<<<compute_norms_blocks, compute_norms_threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), norms.data_ptr<float>(), batch_size, dim);

    int total_elements = batch_size * dim;
    const int normalize_threads = 1024;
    dim3 normalize_blocks((total_elements + normalize_threads - 1) / normalize_threads);
    normalize_kernel<<<normalize_blocks, normalize_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), norms.data_ptr<float>(), out.data_ptr<float>(), batch_size, dim);

    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}
"""

l2_normalize_cpp_source = """
torch::Tensor l2_normalize_cuda(torch::Tensor x);
"""

l2_normalize = load_inline(
    name="l2_normalize",
    cpp_sources=l2_normalize_cpp_source,
    cuda_sources=l2_normalize_source,
    functions=["l2_normalize_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_normalize = l2_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2_normalize.l2_normalize_cuda(x)