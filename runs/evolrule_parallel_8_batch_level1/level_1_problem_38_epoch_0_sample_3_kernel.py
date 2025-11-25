import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void l1_normalize_fused(const float* x, float* out, int batch_size, int dim) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float acc = 0.0f;

    // Compute the sum of absolute values for this row
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = abs(x[row * dim + i]);
        acc += val;
    }

    shared[tid] = acc;
    __syncthreads();

    // Reduce to find the sum_abs
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    float sum_abs = shared[0];

    // Compute the normalization factor
    if (sum_abs == 0) {
        sum_abs = 1e-8; // Avoid division by zero
    }
    float inv_s = dim / sum_abs;

    // Write the normalized values
    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = row * dim + i;
        out[idx] = x[idx] * inv_s;
    }
}

torch::Tensor l1_normalize_fused_cuda(torch::Tensor x) {
    int batch_size = x.size(0);
    int dim = x.size(1);
    auto out = torch::empty_like(x);

    int threadsPerBlock = 256;
    int sharedSize = threadsPerBlock * sizeof(float);

    l1_normalize_fused<<<batch_size, threadsPerBlock, sharedSize>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim
    );

    return out;
}
"""

custom_ops = load_inline(
    name="l1_norm_fused",
    cuda_sources=cuda_source,
    functions=["l1_normalize_fused_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return custom_ops.l1_normalize_fused_cuda(x)