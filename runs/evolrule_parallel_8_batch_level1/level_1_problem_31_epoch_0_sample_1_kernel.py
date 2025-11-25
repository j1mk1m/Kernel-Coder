import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_elu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void elu_forward(const float* x, float* y, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        y[idx] = (xi > 0) ? xi : alpha * (expf(xi) - 1.0f);
    }
}

__global__ void elu_backward(const float* x, const float* grad_out, float* grad_x, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xi = x[idx];
        float grad = (xi > 0) ? 1.0f : alpha * expf(xi);
        grad_x[idx] = grad * grad_out[idx];
    }
}

torch::Tensor elu_forward_cuda(torch::Tensor x, float alpha) {
    const int block_size = 256;
    int n = x.numel();
    auto y = torch::empty_like(x);
    int num_blocks = (n + block_size - 1) / block_size;

    elu_forward<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), alpha, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in forward: %s\\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    return y;
}

std::tuple<torch::Tensor> elu_backward_cuda(torch::Tensor grad_out, torch::Tensor x, float alpha) {
    int n = x.numel();
    auto grad_x = torch::empty_like(x);
    const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    elu_backward<<<num_blocks, block_size>>>(x.data_ptr<float>(), grad_out.data_ptr<float>(), grad_x.data_ptr<float>(), alpha, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in backward: %s\\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    return std::make_tuple(grad_x);
}
"""

elementwise_elu_header = """
torch::Tensor elu_forward_cuda(torch::Tensor x, float alpha);
std::tuple<torch::Tensor> elu_backward_cuda(torch::Tensor grad_out, torch::Tensor x, float alpha);
"""

# Load the CUDA extension
elu_extension = load_inline(
    name="elu_extension",
    cpp_sources=elementwise_elu_header,
    cuda_sources=elementwise_elu_source,
    functions=["elu_forward_cuda", "elu_backward_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_ldflags=[""],
)

class EluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return elu_extension.elu_forward_cuda(x, alpha)

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_x = elu_extension.elu_backward_cuda(grad_out, x, alpha)[0]
        return grad_x, None

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return EluFunction.apply(x, self.alpha)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim).cuda()
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization