import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

rmsnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_sum_squares(const float* x, float* sum_squares, int B, int F, int D1, int D2, float eps) {
    int block_idx = blockIdx.x;
    int b = block_idx / (D1 * D2);
    int rem = block_idx % (D1 * D2);
    int d1 = rem / D2;
    int d2 = rem % D2;

    int f = threadIdx.x;
    int x_offset = b * F * D1 * D2 + f * D1 * D2 + d1 * D2 + d2;

    float val = x[x_offset];
    float square = val * val;

    extern __shared__ float shared_squares[];
    shared_squares[threadIdx.x] = square;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < F; i++) {
            total += shared_squares[i];
        }
        int sum_sq_offset = b * D1 * D2 + d1 * D2 + d2;
        sum_squares[sum_sq_offset] = total;
    }
}

__global__ void compute_output(const float* x, const float* sum_squares, float* output, int B, int F, int D1, int D2, float eps) {
    int block_idx = blockIdx.x;
    int b = block_idx / (D1 * D2);
    int rem = block_idx % (D1 * D2);
    int d1 = rem / D2;
    int d2 = rem % D2;

    int f = threadIdx.x;

    extern __shared__ float denom_shared[];
    int sum_sq_offset = b * D1 * D2 + d1 * D2 + d2;
    float sum_sq = sum_squares[sum_sq_offset];

    if (threadIdx.x == 0) {
        float denom_val = sqrtf( (sum_sq / F) + eps );
        denom_shared[0] = denom_val;
    }
    __syncthreads();

    float denom = denom_shared[0];
    int x_offset = b * F * D1 * D2 + f * D1 * D2 + d1 * D2 + d2;
    output[x_offset] = x[x_offset] / denom;
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, int F, float eps) {
    if (!x.is_cuda()) {
        AT_ERROR("x must be a CUDA tensor");
    }
    if (!x.is_contiguous()) {
        AT_ERROR("x must be contiguous");
    }

    int B = x.size(0);
    int D1 = x.size(2);
    int D2 = x.size(3);

    auto options = x.options();
    auto sum_squares = torch::empty({B, D1, D2}, options);

    dim3 blocks(B * D1 * D2);
    dim3 threads(F);

    int sharedMemSize = F * sizeof(float);
    compute_sum_squares<<<blocks, threads, sharedMemSize>>>(
        x.data_ptr<float>(),
        sum_squares.data_ptr<float>(),
        B, F, D1, D2, eps
    );

    auto output = torch::empty_like(x);

    sharedMemSize = 1 * sizeof(float);
    compute_output<<<blocks, threads, sharedMemSize>>>(
        x.data_ptr<float>(),
        sum_squares.data_ptr<float>(),
        output.data_ptr<float>(),
        B, F, D1, D2, eps
    );

    return output;
}
"""

rmsnorm_cpp_source = (
    "torch::Tensor rmsnorm_cuda(torch::Tensor x, int F, float eps);"
)

rmsnorm = load_inline(
    name="rmsnorm",
    cpp_sources=rmsnorm_cpp_source,
    cuda_sources=rmsnorm_source,
    functions=["rmsnorm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm.rmsnorm_cuda(x, self.num_features, self.eps)