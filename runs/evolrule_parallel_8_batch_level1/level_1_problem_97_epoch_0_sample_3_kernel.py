import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for scaled_dot_product_attention
sdpa_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void scaled_dot_product_attention_forward(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int B, int H, int S, int D
) {
    int batch = blockIdx.x / H;
    int head = blockIdx.x % H;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = threadIdx.x;

    if (row >= S || col >= S) return;

    scalar_t sum = 0.0;
    for (int d = 0; d < D; ++d) {
        sum += q[batch * H * S * D + head * S * D + row * D + d] *
               k[batch * H * S * D + head * S * D + col * D + d];
    }
    sum /= sqrt(D);

    // Softmax computation
    // Compute maximum value for numerical stability
    scalar_t max_val = -FLT_MAX;
    for (int i = 0; i < S; ++i) {
        scalar_t val = q[batch * H * S * D + head * S * D + row * D + i] *
                      k[batch * H * S * D + head * S * D + col * D + i];
        if (val > max_val) max_val = val;
    }

    // Compute exponentials and sum
    scalar_t numerator = exp(sum - max_val);
    scalar_t denominator = 0.0;
    for (int i = 0; i < S; ++i) {
        scalar_t val = q[batch * H * S * D + head * S * D + row * D + i] *
                      k[batch * H * S * D + head * S * D + col * D + i];
        denominator += exp((val / sqrt(D)) - max_val);
    }

    scalar_t weight = numerator / denominator;

    // Update output
    for (int d = 0; d < D; ++d) {
        out[batch * H * S * D + head * S * D + row * D + d] +=
            weight * v[batch * H * S * D + head * S * D + col * D + d];
    }
}

at::Tensor scaled_dot_product_attention_cuda(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v
) {
    const int B = q.size(0);
    const int H = q.size(1);
    const int S = q.size(2);
    const int D = q.size(3);

    at::Tensor out = at::zeros({B, H, S, D}, q.options());

    dim3 threads(S, 1);
    dim3 blocks(B * H, S);

    AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "scaled_dot_product_attention_forward", ([&]{
        scaled_dot_product_attention_forward<scalar_t><<<blocks, threads>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B, H, S, D
        );
    }));

    return out;
}
"""

sdpa = load_inline(
    name="sdpa",
    cpp_sources="",
    cuda_sources=sdpa_source,
    functions=["scaled_dot_product_attention_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.sdpa = sdpa

    def forward(self, Q, K, V):
        return self.sdpa.scaled_dot_product_attention_cuda(Q, K, V)