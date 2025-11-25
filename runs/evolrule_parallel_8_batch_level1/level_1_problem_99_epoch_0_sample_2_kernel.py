import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

triplet_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <int BlockSize>
__global__ void triplet_loss_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* loss,
    int batch_size,
    int dim,
    float margin) {

    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    extern __shared__ float shared_sums[];
    float* sum_ap = &shared_sums[0 * BlockSize];
    float* sum_an = &shared_sums[1 * BlockSize];

    int tid = threadIdx.x;

    sum_ap[tid] = 0.0f;
    sum_an[tid] = 0.0f;

    for (int j = tid; j < dim; j += BlockSize) {
        int idx = sample_idx * dim + j;
        float a = anchor[idx];
        float p = positive[idx];
        float n = negative[idx];

        float diff_ap = a - p;
        sum_ap[tid] += diff_ap * diff_ap;

        float diff_an = a - n;
        sum_an[tid] += diff_an * diff_an;
    }

    __syncthreads();

    for (int s = BlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_ap[tid] += sum_ap[tid + s];
            sum_an[tid] += sum_an[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum_ap = sum_ap[0];
        float total_sum_an = sum_an[0];

        float d_ap = sqrt(total_sum_ap);
        float d_an = sqrt(total_sum_an);

        float loss_i = d_ap - d_an + margin;
        loss[sample_idx] = loss_i > 0 ? loss_i : 0.0f;
    }
}

torch::Tensor triplet_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    const int batch_size = anchor.size(0);
    const int dim = anchor.size(1);

    auto loss = torch::empty({batch_size}, torch::device("cuda"));

    const int block_size = 256;
    int num_blocks = batch_size;

    size_t shared_mem_size = 2 * block_size * sizeof(float);

    triplet_loss_kernel<block_size><<<num_blocks, block_size, shared_mem_size>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        loss.data_ptr<float>(),
        batch_size,
        dim,
        margin);

    cudaDeviceSynchronize();

    return loss.mean();
}
"""

triplet_loss_header = """
torch::Tensor triplet_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin);
"""

triplet_loss_cuda = load_inline(
    name="triplet_loss",
    cpp_sources=triplet_loss_header,
    cuda_sources=triplet_loss_source,
    functions=["triplet_loss_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return triplet_loss_cuda.triplet_loss_cuda(
            anchor, positive, negative, self.margin
        )