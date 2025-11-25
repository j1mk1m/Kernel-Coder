import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code for forward and backward passes
kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Function declarations
std::tuple<torch::Tensor, torch::Tensor> smooth_l1_forward_cuda(
    torch::Tensor predictions,
    torch::Tensor targets);

torch::Tensor smooth_l1_backward_cuda(
    torch::Tensor diff,
    int64_t numel,
    float inv_numel);

// Forward kernel
__global__ void smooth_l1_forward_kernel(
    const float* predictions,
    const float* targets,
    float* diff,
    float* loss_sum,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float d = predictions[idx] - targets[idx];
        diff[idx] = d;
        float abs_d = fabsf(d);
        float loss_i;
        if (abs_d < 1.0f) {
            loss_i = 0.5f * d * d;
        } else {
            loss_i = abs_d - 0.5f;
        }
        atomicAdd(loss_sum, loss_i);
    }
}

// Backward kernel
__global__ void smooth_l1_backward_kernel(
    const float* diff,
    float* grad_out,
    int numel,
    float inv_numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float d = diff[idx];
        float grad;
        if (fabsf(d) < 1.0f) {
            grad = d;
        } else {
            grad = d > 0 ? 1.0f : -1.0f;
        }
        grad *= inv_numel;
        grad_out[idx] = grad;
    }
}

// Implementation of forward function
std::tuple<torch::Tensor, torch::Tensor> smooth_l1_forward_cuda(
    torch::Tensor predictions,
    torch::Tensor targets) {
    // Check inputs
    TORCH_CHECK(predictions.device().is_cuda());
    TORCH_CHECK(targets.device().is_cuda());
    TORCH_CHECK(predictions.sizes() == targets.sizes());

    int64_t numel = predictions.numel();
    auto options = predictions.options();
    auto diff = torch::empty_like(predictions);
    auto loss_sum = torch::zeros(1, options);

    const int block_size = 256;
    int num_blocks = (numel + block_size - 1) / block_size;

    smooth_l1_forward_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        diff.data_ptr<float>(),
        loss_sum.data_ptr<float>(),
        numel
    );

    cudaDeviceSynchronize();

    float mean_loss_val = loss_sum[0].item<float>() / numel;
    auto mean_loss = torch::full({}, mean_loss_val, options);

    return std::make_tuple(mean_loss, diff);
}

// Implementation of backward function
torch::Tensor smooth_l1_backward_cuda(
    torch::Tensor diff,
    int64_t numel,
    float inv_numel) {
    auto grad_out = torch::empty_like(diff);

    const int block_size = 256;
    int num_blocks = (numel + block_size - 1) / block_size;

    smooth_l1_backward_kernel<<<num_blocks, block_size>>>(
        diff.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        numel,
        inv_numel
    );

    cudaDeviceSynchronize();

    return grad_out;
}
"""

# Load the CUDA kernels
custom_ops = load_inline(
    name="custom_smooth_l1",
    cuda_sources=kernel_source,
    functions=[
        "std::tuple<torch::Tensor, torch::Tensor> smooth_l1_forward_cuda(torch::Tensor, torch::Tensor)",
        "torch::Tensor smooth_l1_backward_cuda(torch::Tensor, int64_t, float)",
    ],
    verbose=False,
)

class SmoothL1LossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        mean_loss, diff = custom_ops.smooth_l1_forward_cuda(predictions, targets)
        ctx.save_for_backward(diff)
        return mean_loss

    @staticmethod
    def backward(ctx, grad_output):
        diff, = ctx.saved_tensors
        numel = diff.numel()
        inv_numel = 1.0 / numel
        grad_pred = custom_ops.smooth_l1_backward_cuda(diff, numel, inv_numel)
        grad_pred.mul_(grad_output)
        return grad_pred, None

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return SmoothL1LossFunction.apply(predictions, targets)