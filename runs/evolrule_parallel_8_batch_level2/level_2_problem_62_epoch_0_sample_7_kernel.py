import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused kernel: matrix multiplication + group normalization + leaky ReLU + element-wise addition
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template <typename T>
__global__ void fused_forward(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_groups,
    float eps,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;

    // Compute FC layer: x * weight + bias (without separate bias add)
    int row = idx / hidden_size;
    int col = idx % hidden_size;
    T sum = 0;
    for (int i = 0; i < input_size; ++i) {
        sum += input[row * input_size + i] * weight[i * hidden_size + col];
    }
    sum += bias[col];

    // GroupNorm: compute mean and variance for each group
    int group_size = hidden_size / num_groups;
    int group_id = col / group_size;
    int local_col = col % group_size;

    extern __shared__ T shared[];
    T* local_data = shared;
    local_data[threadIdx.x] = sum;

    __syncthreads();

    T mean = 0;
    for (int i = 0; i < group_size; ++i) {
        mean += local_data[threadIdx.x + i * blockDim.x];
    }
    mean /= group_size;

    T var = 0;
    for (int i = 0; i < group_size; ++i) {
        var += (local_data[threadIdx.x + i * blockDim.x] - mean) * (local_data[threadIdx.x + i * blockDim.x] - mean);
    }
    var = (var / group_size) + eps;
    T std_inv = 1.0 / sqrt(var);

    // Apply group norm
    T norm_val = (local_data[threadIdx.x] - mean) * std_inv;

    // Apply Leaky ReLU
    norm_val = norm_val < 0 ? norm_val * negative_slope : norm_val;

    // Element-wise addition x + x
    norm_val *= 2.0;

    // Apply gamma and beta (scale and shift)
    output[idx] = norm_val * gamma[col] + beta[col];
}

torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps,
    float negative_slope
) {
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int hidden_size = weight.size(0);

    auto output = torch::empty({batch_size, hidden_size}, input.options());

    const int threads = 256;
    const int blocks = (batch_size * hidden_size + threads - 1) / threads;

    // Shared memory size: blockDim.x * sizeof(float)
    size_t sm_size = threads * sizeof(float);

    fused_forward<float><<<blocks, threads, sm_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size,
        num_groups,
        eps,
        negative_slope
    );

    return output;
}
"""

# Define C++ header for compilation
fused_header = """
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps,
    float negative_slope
);
"""

# Compile the fused kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_header,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(ModelNew, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.fused_forward = fused_ops.fused_forward_cuda

    def forward(self, x):
        # Directly call fused kernel instead of separate operations
        return self.fused_forward(
            x,
            self.fc.weight,
            self.fc.bias,
            self.gn.weight,
            self.gn.bias,
            self.gn.num_groups,
            self.gn.eps,
            self.leaky_relu.negative_slope
        )