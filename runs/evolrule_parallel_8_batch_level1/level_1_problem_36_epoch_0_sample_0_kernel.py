import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rms_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(
    const float* x,
    float* out,
    int batch_size,
    int dim1,
    int dim2,
    float eps
) {
    const int Features = 64;

    int b = blockIdx.x;
    int d1 = blockIdx.y;
    int d2 = blockIdx.z;
    int f = threadIdx.x;

    if (b >= batch_size || d1 >= dim1 || d2 >= dim2) {
        return;
    }
    if (f >= Features) {
        return;
    }

    int index = b * Features * dim1 * dim2 + f * dim1 * dim2 + d1 * dim2 + d2;

    float x_val = x[index];
    float x_sq = x_val * x_val;

    __shared__ float shared_squares[64];
    shared_squares[f] = x_sq;
    __syncthreads();

    for (int s = 32; s >= 1; s >>= 1) {
        if (threadIdx.x < s) {
            shared_squares[threadIdx.x] += shared_squares[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum = shared_squares[0];
    float mean = sum / Features;
    float rms = sqrt(mean + eps);

    out[index] = x_val / rms;
}

torch::Tensor rms_norm_cuda(torch::Tensor x, float eps) {
    auto batch_size = x.size(0);
    auto dim1 = x.size(2);
    auto dim2 = x.size(3);

    auto out = torch::empty_like(x);

    dim3 threads(Features);
    dim3 blocks(batch_size, dim1, dim2);

    rms_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        dim1,
        dim2,
        eps
    );

    return out;
}
"""

rms_norm_cpp_source = """
torch::Tensor rms_norm_cuda(torch::Tensor x, float eps);
"""

rms_norm = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_source,
    functions=["rms_norm_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm.rms_norm_cuda(x, self.eps)