import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cross_entropy_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void cross_entropy_loss_forward(
    const scalar_t* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    scalar_t* output,
    int batch_size,
    int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    scalar_t max_val = -INFINITY;
    for (int c = 0; c < num_classes; ++c) {
        scalar_t val = predictions[idx * num_classes + c];
        if (val > max_val) {
            max_val = val;
        }
    }

    scalar_t sum_exp = 0;
    for (int c = 0; c < num_classes; ++c) {
        sum_exp += exp(predictions[idx * num_classes + c] - max_val);
    }

    int target = targets[idx];
    scalar_t log_prob = -max_val - log(sum_exp) + predictions[idx * num_classes + target];

    atomicAdd(output, -log_prob);
}

torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int batch_size = predictions.size(0);
    const int num_classes = predictions.size(1);

    auto output = torch::zeros(1, predictions.options());

    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "cross_entropy_loss", ([&] {
        cross_entropy_loss_forward<scalar_t><<<grid_size, block_size>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_classes);
    }));

    output.div_(batch_size);
    cudaDeviceSynchronize();
    return output;
}
"""

cross_entropy_loss_header = """
torch::Tensor cross_entropy_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

cross_entropy_loss = load_inline(
    name="cross_entropy_loss",
    cpp_sources=cross_entropy_loss_header,
    cuda_sources=cross_entropy_loss_source,
    functions=["cross_entropy_loss_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-DWITH_CUDA"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_func = cross_entropy_loss.cross_entropy_loss_cuda

    def forward(self, predictions, targets):
        return self.forward_func(predictions, targets)

def get_inputs():
    batch_size = 32768
    num_classes = 4096
    predictions = torch.rand(batch_size, num_classes, device='cuda', dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device='cuda', dtype=torch.int64)
    return [predictions, targets]

def get_init_inputs():
    return []