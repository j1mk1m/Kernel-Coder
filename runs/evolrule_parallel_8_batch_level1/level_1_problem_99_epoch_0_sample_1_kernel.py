import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Triplet Margin Loss
triplet_loss_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void triplet_loss_kernel(const scalar_t* __restrict__ anchor,
                                   const scalar_t* __restrict__ positive,
                                   const scalar_t* __restrict__ negative,
                                   scalar_t* __restrict__ loss,
                                   const float margin,
                                   const int batch_size,
                                   const int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    scalar_t ap_dist_sq = 0.0;
    scalar_t an_dist_sq = 0.0;

    // Compute squared distances for each dimension in parallel
    for (int d = 0; d < dim; ++d) {
        scalar_t ap_diff = anchor[idx * dim + d] - positive[idx * dim + d];
        ap_dist_sq += ap_diff * ap_diff;

        scalar_t an_diff = anchor[idx * dim + d] - negative[idx * dim + d];
        an_dist_sq += an_diff * an_diff;
    }

    scalar_t ap_dist = sqrt(ap_dist_sq);
    scalar_t an_dist = sqrt(an_dist_sq);

    scalar_t loss_val = ap_dist - an_dist + margin;
    if (loss_val > 0) {
        atomicAdd(loss, loss_val);
    }
}

torch::Tensor triplet_loss_cuda(torch::Tensor anchor,
                               torch::Tensor positive,
                               torch::Tensor negative,
                               float margin) {
    const int batch_size = anchor.size(0);
    const int dim = anchor.size(1);
    auto loss = torch::zeros(1, device=anchor.device());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(anchor.type(), "triplet_loss_cuda", ([&] {
        triplet_loss_kernel<scalar_t><<<blocks, threads>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            loss.data_ptr<scalar_t>(),
            margin,
            batch_size,
            dim);
    }));

    loss /= static_cast<float>(batch_size);  // Average loss

    return loss;
}

"""

triplet_loss_cpp_source = (
    "torch::Tensor triplet_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);"
)

# Compile the inline CUDA code
triplet_loss = load_inline(
    name="triplet_loss",
    cpp_sources=triplet_loss_cpp_source,
    cuda_sources=triplet_loss_source,
    functions=["triplet_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_loss = triplet_loss

    def forward(self, anchor, positive, negative):
        return self.triplet_loss.triplet_loss_cuda(
            anchor, positive, negative, self.margin
        )