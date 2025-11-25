import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batchnorm_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void batchnorm_forward_kernel(
    const scalar_t* x_data,
    const scalar_t* mean_data,
    const scalar_t* var_data,
    const scalar_t* gamma_data,
    const scalar_t* beta_data,
    scalar_t eps,
    scalar_t* y_data,
    int N, int C, int H, int W
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C * H * W) return;

    int w = index % W;
    int h = (index / W) % H;
    int c = (index / (W * H)) % C;
    int n = index / (C * W * H);

    scalar_t x_val = x_data[index];
    scalar_t mean_val = mean_data[c];
    scalar_t var_val = var_data[c];
    scalar_t gamma_val = gamma_data[c];
    scalar_t beta_val = beta_data[c];

    scalar_t inv_std = 1.0 / sqrt(var_val + eps);
    scalar_t norm = (x_val - mean_val) * inv_std;
    y_data[index] = gamma_val * norm + beta_val;
}

std::tuple<torch::Tensor> batchnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
) {
    auto output = torch::empty_like(x);

    const int threads = 256;
    const int elements = x.numel();
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.type(), "batchnorm_forward_cuda", ([&] {
        batchnorm_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data<scalar_t>(),
            mean.data<scalar_t>(),
            var.data<scalar_t>(),
            gamma.data<scalar_t>(),
            beta.data<scalar_t>(),
            eps,
            output.data<scalar_t>(),
            x.size(0), x.size(1), x.size(2), x.size(3)
        );
    }));

    return output;
}
"""

batchnorm_forward_cpp_source = """
torch::Tensor batchnorm_forward_cuda(
    torch::Tensor x,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor gamma,
    torch::Tensor beta,
    double eps
);
"""

batchnorm_forward = load_inline(
    name="batchnorm_forward",
    cpp_sources=batchnorm_forward_cpp_source,
    cuda_sources=batchnorm_forward_source,
    functions=["batchnorm_forward_cuda"],
    verbose=True,
)

class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mean, var, gamma, beta, eps):
        output = batchnorm_forward.batchnorm_forward_cuda(x, mean, var, gamma, beta, eps)
        ctx.save_for_backward(x, mean, var, gamma, beta)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, mean, var, gamma, beta = ctx.saved_tensors
        eps = ctx.eps

        N, C, H, W = x.size()
        m = N * H * W

        x_mu = x - mean
        inv_std = 1.0 / torch.sqrt(var + eps)
        x_hat = x_mu * inv_std.view(1, C, 1, 1)

        # Compute gradients for gamma and beta
        dgamma = (x_hat * grad_output).sum([0, 2, 3], keepdim=True).squeeze()
        dbeta = grad_output.sum([0, 2, 3], keepdim=True).squeeze()

        # Compute dx_hat
        dx_hat = grad_output * gamma.view(1, C, 1, 1)

        # Compute mean of dx_hat and dx_hat * x_hat
        mean_dx_hat = dx_hat.mean(dim=(0, 2, 3), keepdim=True)
        mean_dx_hat_x_hat = (dx_hat * x_hat).mean(dim=(0, 2, 3), keepdim=True)

        # Compute dx
        dx = (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat) * inv_std.view(1, C, 1, 1)

        return dx, None, None, dgamma, dbeta, None

class ModelNew(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = 1e-5
        self.momentum = 0.1  # default momentum

    def forward(self, x):
        if self.training:
            mean = x.mean([0, 2, 3], keepdim=True)
            var = x.var([0, 2, 3], unbiased=True, keepdim=True)

            with torch.no_grad():
                exponential_average_factor = self.momentum
                self.running_mean = self.running_mean * (1 - exponential_average_factor) + mean.squeeze() * exponential_average_factor
                self.running_var = self.running_var * (1 - exponential_average_factor) + var.squeeze() * exponential_average_factor
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        return BatchNormFunction.apply(x, mean, var, self.weight, self.bias, self.eps)