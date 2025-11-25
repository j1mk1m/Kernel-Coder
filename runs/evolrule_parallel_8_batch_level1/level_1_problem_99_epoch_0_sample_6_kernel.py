import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for Triplet Margin Loss
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_triplet_loss_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ losses,
    int batch_size,
    int embedding_dim,
    float margin) {
    int tid = threadIdx.x;
    int triplet_idx = blockIdx.x;

    if (triplet_idx >= batch_size)
        return;

    extern __shared__ float shared[];
    float* s_ap = shared;
    float* s_an = s_ap + blockDim.x;

    float ap_sum = 0.0f;
    float an_sum = 0.0f;

    for (int i = tid; i < embedding_dim; i += blockDim.x) {
        int idx = triplet_idx * embedding_dim + i;
        float a_val = anchor[idx];
        float p_val = positive[idx];
        float n_val = negative[idx];

        float diff_ap = a_val - p_val;
        float diff_an = a_val - n_val;

        ap_sum += diff_ap * diff_ap;
        an_sum += diff_an * diff_an;
    }

    s_ap[tid] = ap_sum;
    s_an[tid] = an_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_ap[tid] += s_ap[tid + s];
            s_an[tid] += s_an[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float d_ap = sqrtf(s_ap[0]);
        float d_an = sqrtf(s_an[0]);
        float loss_val = d_ap - d_an + margin;
        losses[triplet_idx] = fmaxf(0.0f, loss_val);
    }
}

torch::Tensor triplet_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin) {
    int batch_size = anchor.size(0);
    int embedding_dim = anchor.size(1);

    auto losses = torch::empty({batch_size}, anchor.options());

    const int block_size = 256;
    const int shared_size = block_size * 2 * sizeof(float);

    dim3 blocks(batch_size);
    dim3 threads(block_size);

    compute_triplet_loss_kernel<<<blocks, threads, shared_size>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        losses.data_ptr<float>(),
        batch_size,
        embedding_dim,
        margin
    );

    auto total_loss = losses.sum();
    auto mean_loss = total_loss / batch_size;

    return mean_loss;
}
"""

# C++ declarations for the CUDA function
cpp_source = """
torch::Tensor triplet_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);
"""

# Compile the CUDA extension
triplet_loss = load_inline(
    name="triplet_loss",
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=["triplet_loss_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triplet_loss.triplet_loss_cuda(anchor, positive, negative, self.margin)