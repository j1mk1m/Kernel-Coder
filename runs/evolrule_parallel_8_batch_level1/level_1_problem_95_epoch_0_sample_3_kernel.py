import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cross_entropy_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

template <typename scalar_t>
__global__ void cross_entropy_loss_kernel(
    const scalar_t* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    scalar_t* __restrict__ output,
    int batch_size,
    int num_classes) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    const scalar_t* sample_logits = predictions + tid * num_classes;

    // Compute max logit
    scalar_t max_logit = -INFINITY;
    for (int i = 0; i < num_classes; i += 4) {
        for (int j = 0; j < 4 && i+j < num_classes; ++j) {
            scalar_t logit = sample_logits[i + j];
            if (logit > max_logit) {
                max_logit = logit;
            }
        }
    }

    // Compute sum of exp(logits - max_logit)
    scalar_t sum_exp = 0.0;
    for (int i = 0; i < num_classes; i += 4) {
        for (int j = 0; j < 4 && i+j < num_classes; ++j) {
            scalar_t logit = sample_logits[i + j];
            sum_exp += __expf(logit - max_logit);
        }
    }

    // Compute loss for target class
    int target = targets[tid];
    scalar_t logit_target = sample_logits[target];
    scalar_t log_p = logit_target - max_logit - log(sum_exp);
    output[tid] = -log_p;
}

torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto batch_size = predictions.size(0);
    auto num_classes = predictions.size(1);
    auto output = torch::empty({batch_size}, predictions.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size + block_size - 1) / block_size;
    
    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "cross_entropy_loss_cuda", ([&] {
        cross_entropy_loss_kernel<scalar_t><<<num_blocks, block_size>>>(
            predictions.data<scalar_t>(),
            targets.data<int64_t>(),
            output.data<scalar_t>(),
            batch_size,
            num_classes
        );
    }));
    
    return output.mean();
}
"""

cross_entropy_loss_cpp_source = (
    "torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"
)

cross_entropy_loss = load_inline(
    name="cross_entropy_loss",
    cpp_sources=cross_entropy_loss_cpp_source,
    cuda_sources=cross_entropy_loss_source,
    functions=["cross_entropy_loss_cuda"],
    verbose=True,
    extra_cflags=["-ffast-math", "-DWITH_CUDA"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = cross_entropy_loss

    def forward(self, predictions, targets):
        return self.cross_entropy_loss.cross_entropy_loss_cuda(predictions, targets)