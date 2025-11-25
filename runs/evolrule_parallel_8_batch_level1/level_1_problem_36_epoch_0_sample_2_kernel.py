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
    int features,
    int dim1,
    int dim2,
    float eps
) {
    int block_idx = blockIdx.x;
    int b = block_idx / (dim1 * dim2);
    int rem = block_idx % (dim1 * dim2);
    int d1 = rem / dim2;
    int d2 = rem % dim2;

    int f = threadIdx.x;

    int x_offset = b * features * dim1 * dim2;
    x_offset += f * dim1 * dim2;
    x_offset += d1 * dim2 + d2;

    float val = x[x_offset];
    float squared = val * val;

    __shared__ float shared_squares[1024];

    if (threadIdx.x < features) {
        shared_squares[threadIdx.x] = squared;
    } else {
        shared_squares[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Reduction step
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_squares[threadIdx.x] += shared_squares[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum = shared_squares[0];
    float mean = sum / features;
    float rms = sqrt(mean + eps);

    if (threadIdx.x < features) {
        out[x_offset] = val / rms;
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, int features, int dim1, int dim2, float eps) {
    x = x.contiguous();
    auto output = torch::empty_like(x);

    int batch_size = x.size(0);
    dim3 block_size(features);
    dim3 grid_size(batch_size * dim1 * dim2);

    rms_norm_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        dim1,
        dim2,
        eps
    );

    return output;
}
"""

rms_norm_cpp_source = (
    "torch::Tensor rms_norm_cuda(torch::Tensor x, int features, int dim1, int dim2, float eps);"
)

rms_norm = load_inline(
    name="rms_norm",
    cpp_sources=rms_norm_cpp_source,
    cuda_sources=rms_norm_source,
    functions=["rms_norm_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm.rms_norm_cuda(
            x, self.num_features, dim1=512, dim2=512, eps=self.eps
        )