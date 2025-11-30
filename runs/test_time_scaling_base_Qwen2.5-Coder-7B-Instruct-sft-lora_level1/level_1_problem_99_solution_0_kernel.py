import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Triplet Margin Loss
triplet_margin_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom implementation of Triplet Margin Loss
__global__ void triplet_margin_loss_kernel(const float* anchor, const float* positive, const float* negative, float* loss, int batch_size, float margin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float pos_dist = __fadd_rn(__fsub_rn(anchor[idx], positive[idx]), __fsub_rn(anchor[idx], positive[idx]));
        float neg_dist = __fadd_rn(__fsub_rn(anchor[idx], negative[idx]), __fsub_rn(anchor[idx], negative[idx]));
        float dist_diff = __fsub_rn(pos_dist, neg_dist);
        float term = __fmax_rn(-dist_diff, 0.0f);
        loss[idx] = __fadd_rn(term, margin);
    }
}

torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin) {
    auto batch_size = anchor.size(0);
    auto loss = torch::zeros(batch_size, dtype=torch.float32, device=anchor.device);

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    triplet_margin_loss_kernel<<<num_blocks, block_size>>>(anchor.data_ptr<float>(), positive.data_ptr<float>(), negative.data_ptr<float>(), loss.data_ptr<float>(), batch_size, margin);

    return loss.sum();
}
"""

triplet_margin_loss_cpp_source = (
    "torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);"
)

# Compile the inline CUDA code for Triplet Margin Loss
triplet_margin_loss = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_margin_loss_cpp_source,
    cuda_sources=triplet_margin_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.triplet_margin_loss = triplet_margin_loss

    def forward(self, anchor, positive, negative):
        return self.triplet_margin_loss.triplet_margin_loss_cuda(anchor, positive, negative, self.margin)