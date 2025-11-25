import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for KL Divergence
kl_div_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void kl_div_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2> predictions,
    const torch::PackedTensorAccessor<scalar_t,2> targets,
    torch::PackedTensorAccessor<scalar_t,1> output,
    int batch_size,
    int dim_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    scalar_t sum = 0;
    for (int d = 0; d < dim_size; ++d) {
        scalar_t p = predictions[idx][d];
        scalar_t t = targets[idx][d];
        if (t > 1e-20) {  // Avoid log(0) and 0 * log(0)
            if (p > 1e-20) {
                sum += p * log(p / t);
            }
        } else {
            // If target is zero, contribution is zero only if prediction is also zero
            if (p > 1e-20) {
                sum += p * log(p / 1e-20);  // Avoid division by zero, but this might need adjustment
            }
        }
    }
    output[idx] = sum / dim_size;
}

template <typename scalar_t>
__global__ void kl_div_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2> predictions,
    const torch::PackedTensorAccessor<scalar_t,2> targets,
    torch::PackedTensorAccessor<scalar_t,2> grad_predictions,
    torch::PackedTensorAccessor<scalar_t,1> grad_output,
    int batch_size,
    int dim_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    scalar_t grad_scale = grad_output[idx] / dim_size;
    for (int d = 0; d < dim_size; ++d) {
        scalar_t p = predictions[idx][d];
        scalar_t t = targets[idx][d];
        if (t > 1e-20) {
            if (p > 1e-20) {
                grad_predictions[idx][d] = grad_scale * (log(p / t) + 1);
            } else {
                grad_predictions[idx][d] = 0;
            }
        } else {
            if (p > 1e-20) {
                grad_predictions[idx][d] = grad_scale * (log(p / 1e-20) + 1);
            } else {
                grad_predictions[idx][d] = 0;
            }
        }
    }
}

#define CHECK_INPUT(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
torch::Tensor kl_div_forward_cuda(
    torch::Tensor predictions,
    torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int batch_size = predictions.size(0);
    int dim_size = predictions.size(1);

    auto output = torch::empty({batch_size}, predictions.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(predictions.type(), "kl_div_forward_cuda", ([&] {
        using scalar_t = scalar_t;
        kl_div_forward_kernel<scalar_t><<<blocks, threads>>>(
            predictions.packed_accessor<scalar_t,2>(),
            targets.packed_accessor<scalar_t,2>(),
            output.packed_accessor<scalar_t,1>(),
            batch_size,
            dim_size);
    }));

    return output;
}

std::tuple<torch::Tensor, torch::Tensor> kl_div_backward_cuda(
    torch::Tensor predictions,
    torch::Tensor targets,
    torch::Tensor grad_output) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);
    CHECK_INPUT(grad_output);

    int batch_size = predictions.size(0);
    int dim_size = predictions.size(1);

    auto grad_predictions = torch::zeros_like(predictions);
    auto grad_targets = torch::zeros_like(targets);  // Not used, but required for API

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(predictions.type(), "kl_div_backward_cuda", ([&] {
        using scalar_t = scalar_t;
        kl_div_backward_kernel<scalar_t><<<blocks, threads>>>(
            predictions.packed_accessor<scalar_t,2>(),
            targets.packed_accessor<scalar_t,2>(),
            grad_predictions.packed_accessor<scalar_t,2>(),
            grad_output.packed_accessor<scalar_t,1>(),
            batch_size,
            dim_size);
    }));

    return std::make_tuple(grad_predictions, grad_targets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_forward_cuda, "KL Divergence forward");
    m.def("backward", &kl_div_backward_cuda, "KL Divergence backward");
}
"""

# Compile the CUDA extension
kl_div = load_inline(
    name="kl_div",
    cpp_sources="",
    cuda_sources=kl_div_source,
    functions=["forward", "backward"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div = kl_div

    def forward(self, predictions, targets):
        return self.kl_div.forward(predictions, targets)

# Backward pass is handled by the autograd engine using the custom backward function
def kl_div_loss_backward_hook(ctx, predictions, targets, grad_output):
    grad_predictions, _ = ctx.kl_div.backward(predictions, targets, grad_output)
    return grad_predictions, None  # No gradient w.r.t targets

class KLDivLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        ctx.kl_div = kl_div
        return kl_div.forward(predictions, targets)

    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        return kl_div_loss_backward_hook(ctx, predictions, targets, grad_output), None

# Overriding the forward method to use the custom autograd function
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_loss_func = KLDivLossFunction.apply

    def forward(self, predictions, targets):
        return self.kl_loss_func(predictions, targets)