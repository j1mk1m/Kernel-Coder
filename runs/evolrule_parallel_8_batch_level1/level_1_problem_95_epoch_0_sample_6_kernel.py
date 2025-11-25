import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cross_entropy_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_BLOCK 1024
#define CUDA_1D_GRID(n) ((n + CUDA_1D_BLOCK - 1) / CUDA_1D_BLOCK)

template <typename scalar_t>
__global__ void log_softmax_forward_kernel(const scalar_t* input, scalar_t* output, int batch_size, int classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        scalar_t max_val = input[idx * classes];
        for (int c = 1; c < classes; ++c) {
            if (input[idx * classes + c] > max_val) {
                max_val = input[idx * classes + c];
            }
        }
        scalar_t sum = 0;
        for (int c = 0; c < classes; ++c) {
            sum += exp(input[idx * classes + c] - max_val);
        }
        scalar_t inv_sum = 1.0 / sum;
        for (int c = 0; c < classes; ++c) {
            output[idx * classes + c] = input[idx * classes + c] - max_val - log(sum);
        }
    }
}

template <typename scalar_t>
__global__ void nll_loss_forward_kernel(const scalar_t* log_probs, scalar_t* output, const int64_t* targets, int batch_size, int classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int target = targets[idx];
        if (target < 0 || target >= classes) {
            return; // ignore invalid targets
        }
        atomicAdd(output, -log_probs[idx * classes + target]);
    }
}

torch::Tensor cross_entropy_cuda(torch::Tensor input, torch::Tensor targets) {
    auto batch_size = input.size(0);
    auto classes = input.size(1);
    auto log_probs = torch::empty_like(input);
    auto loss_sum = torch::zeros(1, device=input.device());

    const dim3 blocks(CUDA_1D_GRID(batch_size));
    const dim3 threads(CUDA_1D_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward", ([&]{
        log_softmax_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), log_probs.data_ptr<scalar_t>(), batch_size, classes);
    }));

    blocks = dim3(CUDA_1D_GRID(batch_size));
    threads = dim3(CUDA_1D_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "nll_loss_forward", ([&]{
        nll_loss_forward_kernel<scalar_t><<<blocks, threads>>>(
            log_probs.data_ptr<scalar_t>(), loss_sum.data_ptr<scalar_t>(), targets.data_ptr<int64_t>(), batch_size, classes);
    }));

    return loss_sum / batch_size;
}
"""

cross_entropy_cpp_source = (
    "torch::Tensor cross_entropy_cuda(torch::Tensor input, torch::Tensor targets);"
)

cross_entropy = load_inline(
    name="cross_entropy_cuda",
    cpp_sources=cross_entropy_cpp_source,
    cuda_sources=cross_entropy_source,
    functions=["cross_entropy_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cross_entropy = cross_entropy

    def forward(self, predictions, targets):
        return self.cross_entropy.cross_entropy_cuda(predictions, targets.long())