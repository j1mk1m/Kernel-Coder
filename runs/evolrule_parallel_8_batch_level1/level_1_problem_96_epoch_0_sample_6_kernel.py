import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Smooth L1 Loss
smooth_l1_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* loss,
    int batch_size,
    int dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < batch_size * dim; i += stride) {
        const int batch = i / dim;
        const int feature = i % dim;

        const float diff = predictions[i] - targets[i];
        const float abs_diff = fabs(diff);
        float loss_val = (abs_diff < 1.0f) ? 0.5f * diff * diff : abs_diff - 0.5f;

        // Accumulate into the loss array (reduction over features)
        loss[batch] += loss_val;
    }
}

torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets,
    int reduction_dim
) {
    int batch_size = predictions.size(0);
    int dim = predictions.size(1);

    torch::Tensor loss = torch::zeros(batch_size, predictions.options());

    const int block_size = 256;
    const int num_elements = batch_size * dim;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    smooth_l1_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        loss.data_ptr<float>(),
        batch_size,
        dim
    );

    // Reduce over the batch dimension if required (default is mean)
    if (reduction_dim == 0) {
        torch::Tensor final_loss = loss.mean();
        return final_loss;
    } else if (reduction_dim == 1) {
        return loss;
    } else {
        return loss.sum();
    }
}
"""

cpp_source = "torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, int reduction_dim);"

# Compile the CUDA kernel
smooth_l1_loss = load_inline(
    name="smooth_l1_loss",
    cpp_sources=cpp_source,
    cuda_sources=smooth_l1_loss_source,
    functions=["smooth_l1_loss_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1_loss = smooth_l1_loss

    def forward(self, predictions, targets):
        # PyTorch's default smooth_l1_loss uses mean reduction over the batch
        return self.smooth_l1_loss.smooth_l1_loss_cuda(
            predictions, targets, reduction_dim=0
        )

# Compatibility function to handle input shapes
def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    scale = torch.rand(())
    return [
        torch.rand(batch_size, *input_shape).cuda() * scale,
        torch.rand(batch_size, *input_shape).cuda(),
    ]

def get_init_inputs():
    return []