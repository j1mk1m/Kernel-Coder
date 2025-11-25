import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batchnorm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_mean_var_kernel(
    const float* input,
    float* sum,
    float* sum_squares,
    int batch_size,
    int features,
    int dim1,
    int dim2) {
    int c = blockIdx.x;
    if (c >= features) return;
    int total = batch_size * dim1 * dim2;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float local_sum = 0.0f;
    float local_squares = 0.0f;
    for (int i = tid; i < total; i += stride) {
        int b = i / (dim1 * dim2);
        int rem = i % (dim1 * dim2);
        int h = rem / dim2;
        int w = rem % dim2;
        int offset = b * features * dim1 * dim2 + c * dim1 * dim2 + h * dim2 + w;
        float x = input[offset];
        local_sum += x;
        local_squares += x * x;
    }
    __shared__ float s_sum[256];
    __shared__ float s_squares[256];
    s_sum[threadIdx.x] = local_sum;
    s_squares[threadIdx.x] = local_squares;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_squares[threadIdx.x] += s_squares[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sum[c] = s_sum[0];
        sum_squares[c] = s_squares[0];
    }
}

__global__ void apply_batchnorm_kernel(
    const float* input,
    float* output,
    const float* mean,
    const float* var,
    const float* gamma,
    const float* beta,
    float eps,
    int batch_size,
    int features,
    int dim1,
    int dim2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features * dim1 * dim2) return;
    int c = (idx / (dim1 * dim2)) % features;
    int b = idx / (features * dim1 * dim2);
    int rem = idx % (features * dim1 * dim2);
    int h = (rem / dim2) % dim1;
    int w = rem % dim2;
    float x = input[idx];
    float inv_std = 1.0f / sqrtf(var[c] + eps);
    float normalized = (x - mean[c]) * inv_std;
    float scaled = normalized * gamma[c] + beta[c];
    output[idx] = scaled;
}

std::tuple<torch::Tensor, torch::Tensor> compute_mean_var(
    torch::Tensor input,
    int batch_size,
    int features,
    int dim1,
    int dim2) {
    auto sum = torch::zeros({features}, input.options());
    auto sum_squares = torch::zeros({features}, input.options());
    int block_size = 256;
    dim3 blocks(features);
    dim3 threads(block_size);
    compute_mean_var_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        sum.data_ptr<float>(),
        sum_squares.data_ptr<float>(),
        batch_size, features, dim1, dim2);
    return std::make_tuple(sum, sum_squares);
}

torch::Tensor apply_batchnorm(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor var,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps,
    int batch_size,
    int features,
    int dim1,
    int dim2) {
    auto output = torch::empty_like(input);
    int total = batch_size * features * dim1 * dim2;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    apply_batchnorm_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        eps,
        batch_size, features, dim1, dim2);
    return output;
}

TORCH_LIBRARY_IMPL(inline_ops, CUDA, m) {
    m.def("compute_mean_var", compute_mean_var);
    m.def("apply_batchnorm", apply_batchnorm);
}
"""

# Load the CUDA functions
batchnorm = load_inline(
    name="batchnorm",
    cuda_sources=batchnorm_cuda_source,
    functions=["compute_mean_var", "apply_batchnorm"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5  # Use PyTorch's default epsilon
        self.compute_mean_var = batchnorm.compute_mean_var
        self.apply_batchnorm = batchnorm.apply_batchnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        features = self.weight.size(0)
        dim1 = x.size(2)
        dim2 = x.size(3)
        # Compute sum and sum_squares using CUDA kernel
        sum, sum_squares = self.compute_mean_var(x, batch_size, features, dim1, dim2)
        # Calculate mean and variance
        N = batch_size * dim1 * dim2
        mean = sum / N
        var = sum_squares / N - mean * mean
        # Apply batch normalization using the custom kernel
        return self.apply_batchnorm(x, mean, var, self.weight, self.bias, self.eps, batch_size, features, dim1, dim2)

# Ensure get_inputs and get_init_inputs are compatible
def get_inputs():
    # Original get_inputs uses batch_size=64, features=64, dim1=512, dim2=512
    x = torch.rand(64, 64, 512, 512, device='cuda')
    return [x]

def get_init_inputs():
    # The original model's __init__ takes 'num_features', which is features=64 in get_init_inputs returns [features]
    return [64]