import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for cross entropy loss
cross_entropy_loss_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void CrossEntropyForwardKernel(
    const scalar_t* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    scalar_t* __restrict__ loss_data,
    int batch_size,
    int num_classes) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    const scalar_t* pred = predictions + tid * num_classes;
    int target = targets[tid];

    // Compute the logit for the target class
    scalar_t target_logit = pred[target];

    // Compute max for numerical stability
    scalar_t max_val = pred[0];
    for (int c = 1; c < num_classes; c++) {
        if (pred[c] > max_val) {
            max_val = pred[c];
        }
    }

    // Compute sum of exp(pred[c] - max_val)
    scalar_t sum_exp = 0.0;
    for (int c = 0; c < num_classes; c++) {
        sum_exp += expf(pred[c] - max_val);
    }

    // Compute loss for this sample
    loss_data[tid] = - (target_logit - max_val) + logf(sum_exp);
}

torch::Tensor cross_entropy_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets) {
    const int batch_size = predictions.size(0);
    const int num_classes = predictions.size(1);

    auto loss = torch::empty({batch_size}, predictions.dtype(), predictions.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "cross_entropy_loss_cuda", ([&] {
        CrossEntropyForwardKernel<scalar_t><<<blocks, threads>>>(
            predictions.data<scalar_t>(),
            targets.data<int64_t>(),
            loss.data<scalar_t>(),
            batch_size,
            num_classes);
    }));

    cudaDeviceSynchronize();
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

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda(), torch.randint(0, num_classes, (batch_size,)).cuda()]

def get_init_inputs():
    return []