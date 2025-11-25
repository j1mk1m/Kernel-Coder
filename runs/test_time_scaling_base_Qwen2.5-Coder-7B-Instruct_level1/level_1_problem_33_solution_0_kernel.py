import torch
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for batch normalization
bn_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bn_forward_kernel(const float* x, const float* running_mean, const float* running_var, const float* weight, const float* bias, float* y, float eps, int N, int C, int H, int W) {
    // Implement the forward pass of batch normalization here
}

__global__ void bn_backward_kernel(const float* x, const float* running_mean, const float* running_var, const float* weight, const float* bias, const float* dy, float* dx, float* dw, float* db, float eps, int N, int C, int H, int W) {
    // Implement the backward pass of batch normalization here
}

void bn_forward_cuda(torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, torch::Tensor y, float eps) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    bn_forward_kernel<<<N, C>>>(x.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), eps, N, C, H, W);
}

void bn_backward_cuda(torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, torch::Tensor dy, torch::Tensor dx, torch::Tensor dw, torch::Tensor db, float eps) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    bn_backward_kernel<<<N, C>>>(x.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), dy.data_ptr<float>(), dx.data_ptr<float>(), dw.data_ptr<float>(), db.data_ptr<float>(), eps, N, C, H, W);
}
"""

bn_cpp_source = (
    "void bn_forward_cuda(torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, torch::Tensor y, float eps);"
    "void bn_backward_cuda(torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, torch::Tensor dy, torch::Tensor dx, torch::Tensor dw, torch::Tensor db, float eps);"
)

# Compile the inline CUDA code for batch normalization
bn = load_inline(
    name="bn",
    cpp_sources=bn_cpp_source,
    cuda_sources=bn_source,
    functions=["bn_forward_cuda", "bn_backward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.running_mean = torch.zeros(num_features).cuda()
        self.running_var = torch.ones(num_features).cuda()
        self.weight = torch.ones(num_features).cuda()
        self.bias = torch.zeros(num_features).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x).cuda()
        bn_forward_cuda(x, self.running_mean, self.running_var, self.weight, self.bias, y, 1e-5)
        return y

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        dx = torch.zeros_like(grad_output).cuda()
        dw = torch.zeros_like(self.weight).cuda()
        db = torch.zeros_like(self.bias).cuda()
        bn_backward_cuda(grad_output, self.running_mean, self.running_var, self.weight, self.bias, grad_output, dx, dw, db, 1e-5)
        return dx, dw, db