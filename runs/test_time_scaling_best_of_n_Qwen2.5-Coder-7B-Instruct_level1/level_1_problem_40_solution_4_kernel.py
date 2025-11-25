import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Layer Normalization
layer_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layer_norm_forward_kernel(const float* input, float* mean, float* var, float* output, int batch_size, int features, int dim1, int dim2) {
    // TODO: Implement the forward pass of Layer Normalization here
}

__global__ void layer_norm_backward_kernel(const float* grad_output, const float* input, const float* mean, const float* var, float* grad_input, float* grad_gamma, float* grad_beta, int batch_size, int features, int dim1, int dim2) {
    // TODO: Implement the backward pass of Layer Normalization here
}

torch::Tensor layer_norm_forward_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta) {
    auto batch_size = input.size(0);
    auto features = input.size(1);
    auto dim1 = input.size(2);
    auto dim2 = input.size(3);
    auto mean = torch::zeros({batch_size, features}, device=input.device());
    auto var = torch::zeros({batch_size, features}, device=input.device());

    const int block_size = 256;
    const int num_blocks = (batch_size * features + block_size - 1) / block_size;

    layer_norm_forward_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), output.data_ptr<float>(), batch_size, features, dim1, dim2);

    return output;
}

torch::Tensor layer_norm_backward_cuda(torch::Tensor grad_output, torch::Tensor input, torch::Tensor gamma, torch::Tensor beta) {
    auto batch_size = input.size(0);
    auto features = input.size(1);
    auto dim1 = input.size(2);
    auto dim2 = input.size(3);
    auto mean = torch::zeros({batch_size, features}, device=input.device());
    auto var = torch::zeros({batch_size, features}, device=input.device());
    auto grad_input = torch::zeros_like(input);
    auto grad_gamma = torch::zeros_like(gamma);
    auto grad_beta = torch::zeros_like(beta);

    const int block_size = 256;
    const int num_blocks = (batch_size * features + block_size - 1) / block_size;

    layer_norm_backward_kernel<<<num_blocks, block_size>>>(grad_output.data_ptr<float>(), input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), grad_input.data_ptr<float>(), grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(), batch_size, features, dim1, dim2);

    return grad_input;
}
"""

layer_norm_cpp_source = (
    "torch::Tensor layer_norm_forward_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta);"
    "torch::Tensor layer_norm_backward_cuda(torch::Tensor grad_output, torch::Tensor input, torch::Tensor gamma, torch::Tensor beta);"
)

# Compile the inline CUDA code for Layer Normalization
layer_norm = load_inline(
    name="layer_norm",
    cpp_sources=layer_norm_cpp_source,
    cuda_sources=layer_norm_source,
    functions=["layer_norm_forward_cuda", "layer_norm_backward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm.layer_norm_forward_cuda(x, self.gamma, self.beta)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        return layer_norm.layer_norm_backward_cuda(grad_output, x, self.gamma, self.beta)