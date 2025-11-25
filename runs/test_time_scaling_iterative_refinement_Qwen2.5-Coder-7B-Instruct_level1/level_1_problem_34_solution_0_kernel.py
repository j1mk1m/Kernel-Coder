import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instance_norm_forward_kernel(const float* input, float* mean, float* var, float* output, int batch_size, int channels, int height, int width) {
    // Implement the Instance Normalization forward pass here
    // This should compute the mean and variance along the spatial dimensions
    // and then normalize the input using these statistics
}

__global__ void instance_norm_backward_kernel(const float* grad_output, const float* input, const float* mean, const float* var, float* grad_input, float* grad_mean, float* grad_var, int batch_size, int channels, int height, int width) {
    // Implement the Instance Normalization backward pass here
    // This should compute the gradients w.r.t. the input, mean, and variance
}

torch::Tensor instance_norm_forward_cuda(torch::Tensor input) {
    // Prepare buffers for mean, variance, and output
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto output = torch::zeros_like(input);
    auto mean = torch::zeros({batch_size, channels}, input.options().dtype(torch::kFloat32));
    auto var = torch::zeros({batch_size, channels}, input.options().dtype(torch::kFloat32));

    // Launch the forward kernel
    instance_norm_forward_kernel<<<...>>>(input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), output.data_ptr<float>(), batch_size, channels, height, width);

    return output;
}

torch::Tensor instance_norm_backward_cuda(torch::Tensor grad_output, torch::Tensor input, torch::Tensor mean, torch::Tensor var) {
    // Prepare buffers for gradients
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto grad_input = torch::zeros_like(input);
    auto grad_mean = torch::zeros_like(mean);
    auto grad_var = torch::zeros_like(var);

    // Launch the backward kernel
    instance_norm_backward_kernel<<<...>>>(grad_output.data_ptr<float>(), input.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), grad_input.data_ptr<float>(), grad_mean.data_ptr<float>(), grad_var.data_ptr<float>(), batch_size, channels, height, width);

    return grad_input;
}
"""

instance_norm_cpp_source = (
    "torch::Tensor instance_norm_forward_cuda(torch::Tensor input);"
    "torch::Tensor instance_norm_backward_cuda(torch::Tensor grad_output, torch::Tensor input, torch::Tensor mean, torch::Tensor var);"
)

# Compile the inline CUDA code for Instance Normalization
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_forward_cuda", "instance_norm_backward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.instance_norm = instance_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm.instance_norm_forward_cuda(x)

    def backward(self, grad_output: torch.Tensor, input: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return self.instance_norm.instance_norm_backward_cuda(grad_output, input, mean, var)