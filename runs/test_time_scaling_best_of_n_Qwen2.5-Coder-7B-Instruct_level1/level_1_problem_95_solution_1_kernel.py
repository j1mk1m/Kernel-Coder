import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cross entropy loss
cross_entropy_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom implementation of cross entropy loss
__global__ void cross_entropy_loss_kernel(const float* predictions, const int* targets, float* loss, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int target_class = targets[idx];
        float prediction_for_target_class = predictions[idx * num_classes + target_class];
        loss[idx] = -logf(prediction_for_target_class);
    }
}

torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto num_classes = predictions.size(1);
    auto loss = torch::zeros(batch_size, torch::kFloat32);

    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;

    cross_entropy_loss_kernel<<<num_blocks, block_size>>>(predictions.data_ptr<float>(), targets.data_ptr<int>(), loss.data_ptr<float>(), batch_size, num_classes);

    return loss.mean();
}
"""

cross_entropy_loss_cpp_source = (
    "torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

# Compile the inline CUDA code for cross entropy loss
cross_entropy_loss = load_inline(
    name="cross_entropy_loss",
    cpp_sources=cross_entropy_loss_cpp_source,
    cuda_sources=cross_entropy_loss_source,
    functions=["cross_entropy_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy_loss = cross_entropy_loss

    def forward(self, predictions, targets):
        return self.cross_entropy_loss.cross_entropy_loss_cuda(predictions, targets)