import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void compute_terms_kernel(
    const scalar_t* predictions,
    const scalar_t* targets,
    scalar_t* terms,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        scalar_t p = predictions[idx];
        scalar_t q = targets[idx];
        scalar_t log_p = log(static_cast<float>(p));
        scalar_t log_q = log(static_cast<float>(q));
        terms[idx] = p * (log_p - log_q);
    }
}

torch::Tensor compute_terms_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto predictions_t = predictions.contiguous();
    auto targets_t = targets.contiguous();
    auto terms = torch::empty_like(predictions);

    int num_elements = predictions.numel();
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "compute_terms_cuda", ([&] {
        compute_terms_kernel<scalar_t><<<num_blocks, block_size>>>(
            predictions_t.data_ptr<scalar_t>(),
            targets_t.data_ptr<scalar_t>(),
            terms.data_ptr<scalar_t>(),
            num_elements
        );
    }));

    return terms;
}

template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t* predictions,
    const scalar_t* targets,
    const scalar_t* grad_output,
    scalar_t* grad_predictions,
    scalar_t* grad_targets,
    int num_elements,
    scalar_t inv_N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        scalar_t p = predictions[idx];
        scalar_t q = targets[idx];
        scalar_t log_p = log(static_cast<float>(p));
        scalar_t log_q = log(static_cast<float>(q));
        scalar_t go = grad_output[0]; // grad_output is a scalar
        grad_predictions[idx] = go * (log_p - log_q + 1.0) * inv_N;
        grad_targets[idx] = go * (-p / q) * inv_N;
    }
}

void backward_cuda(
    torch::Tensor predictions,
    torch::Tensor targets,
    torch::Tensor grad_output,
    torch::Tensor grad_predictions,
    torch::Tensor grad_targets,
    float inv_N
) {
    int num_elements = predictions.numel();
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "backward_cuda", ([&] {
        backward_kernel<scalar_t><<<num_blocks, block_size>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_predictions.data_ptr<scalar_t>(),
            grad_targets.data_ptr<scalar_t>(),
            num_elements,
            static_cast<scalar_t>(inv_N)
        );
    }));
}
"""

cuda_cpp_sources = """
torch::Tensor compute_terms_cuda(torch::Tensor predictions, torch::Tensor targets);
void backward_cuda(torch::Tensor predictions, torch::Tensor targets, torch::Tensor grad_output, torch::Tensor grad_predictions, torch::Tensor grad_targets, float inv_N);
"""

module = load_inline(
    name="custom_kl_div",
    cpp_sources=cuda_cpp_sources,
    cuda_sources=cuda_source,
    functions=["compute_terms_cuda", "backward_cuda"],
    verbose=True,
)

class CustomKLDivFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        C = predictions.size(-1)
        N = predictions.numel() // C
        inv_N = 1.0 / N
        ctx.save_for_backward(predictions, targets)
        ctx.inv_N = inv_N

        terms = module.compute_terms_cuda(predictions, targets)
        per_sample_sums = terms.view(-1, C).sum(dim=1)
        total_sum = per_sample_sums.sum()
        loss = total_sum / N
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        inv_N = ctx.inv_N

        grad_predictions = torch.empty_like(predictions)
        grad_targets = torch.empty_like(targets)

        module.backward_cuda(
            predictions,
            targets,
            grad_output.view(1),  # Ensure it's a tensor
            grad_predictions,
            grad_targets,
            inv_N
        )

        return grad_predictions, grad_targets

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return CustomKLDivFunction.apply(predictions, targets)