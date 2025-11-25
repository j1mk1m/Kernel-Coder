import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Triplet Margin Loss
triplet_margin_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Implement the Triplet Margin Loss function here using CUDA
__global__ void triplet_margin_loss_kernel(const float* anchor, const float* positive, const float* negative, float* loss, int batch_size, float margin) {
    // Your implementation goes here
}

torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin) {
    auto batch_size = anchor.size(0);
    auto loss = torch::zeros({batch_size}, torch::kFloat32);

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    triplet_margin_loss_kernel<<<num_blocks, block_size>>>(anchor.data_ptr<float>(), positive.data_ptr<float>(), negative.data_ptr<float>(), loss.data_ptr<float>(), batch_size, margin);

    return loss.sum() / batch_size;
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