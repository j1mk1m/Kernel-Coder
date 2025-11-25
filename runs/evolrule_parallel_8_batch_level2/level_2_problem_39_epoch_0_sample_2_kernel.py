import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

template <typename T>
__global__ void fused_forward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ scale,
    T* __restrict__ output,
    T* __restrict__ running_mean,
    T* __restrict__ running_var,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    const T eps,
    const T momentum,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    // Each threadblock handles one output feature channel
    const int channel = blockIdx.x;
    if (channel >= out_features) return;

    // Shared memory for partial sums
    extern __shared__ typename cub::BlockReduce<T, 1024>::TempStorage temp_storage[];
    using BlockReduce = cub::BlockReduce<T, 1024>;

    // Thread indices
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    T sum = 0;
    T sum_sq = 0;

    // Iterate over the batch dimension
    for (int batch = 0; batch < batch_size; batch += total_threads) {
        int idx = batch + tid;
        if (idx >= batch_size) continue;

        // Compute GEMM for this channel (without bias yet)
        T val = 0;
        for (int i = 0; i < in_features; ++i) {
            val += input[batch * in_features + i] * weight[channel * in_features + i];
        }
        val += bias[channel];  // Add bias

        val *= scale[channel];  // Apply scaling

        // Accumulate sum and sum of squares for mean/var computation
        sum += val;
        sum_sq += val * val;
    }

    // Reduce within the block to compute per-channel sums
    T block_sum = BlockReduce(temp_storage[blockIdx.x]).Reduce(sum, cub::Sum());
    T block_sum_sq = BlockReduce(temp_storage[blockIdx.x]).Reduce(sum_sq, cub::Sum());

    if (tid == 0) {
        atomicAdd(&sum, block_sum);
        atomicAdd(&sum_sq, block_sum_sq);
    }
    __syncthreads();

    // Wait for all threads to finish the reductions
    __shared__ T shared_sum;
    __shared__ T shared_sum_sq;
    if (tid == 0) {
        shared_sum = sum;
        shared_sum_sq = sum_sq;
    }
    __syncthreads();

    // Compute mean and variance for this channel
    T mean = shared_sum / batch_size;
    T var = shared_sum_sq / batch_size - mean * mean;
    var = fmax(var, T(0));  // Ensure variance is non-negative

    // Update running statistics
    if (tid == 0) {
        *running_mean = momentum * mean + (1 - momentum) * (*running_mean);
        *running_var = momentum * var + (1 - momentum) * (*running_var);
    }

    // Normalize and apply gamma/beta
    T inv_std = 1.0 / sqrt(var + eps);
    T gamma_val = gamma[channel];
    T beta_val = beta[channel];

    // Second pass: compute output values
    for (int batch = 0; batch < batch_size; batch += total_threads) {
        int idx = batch + tid;
        if (idx >= batch_size) continue;

        T val = 0;
        for (int i = 0; i < in_features; ++i) {
            val += input[idx * in_features + i] * weight[channel * in_features + i];
        }
        val += bias[channel];
        val *= scale[channel];

        // Apply batch norm
        val = (val - mean) * inv_std * gamma_val + beta_val;
        output[idx * out_features + channel] = val;
    }
}

// CUDA wrapper function
torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps,
    float momentum
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    dim3 blocks(out_features);  // One block per output feature
    dim3 threads(1024);         // Use full warp for parallelism

    // Shared memory for CUB reduction (each block needs storage)
    size_t shared_mem_size = sizeof(cub::BlockReduce<float, 1024>::TempStorage) * out_features;
    shared_mem_size = (shared_mem_size + 512) & ~512; // Align to 512 bytes

    fused_forward_kernel<float><<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        output.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        eps,
        momentum,
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_kernel_cpp_source = (
    "torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor scale, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor gamma, torch::Tensor beta, float eps, float momentum);"
)

# Compile the fused kernel
fused_forward = load_inline(
    name="fused_forward",
    cpp_sources=fused_kernel_cpp_source,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward_cuda"],
    verbose=True,
    extra_cflags=["-I/usr/local/cuda/include"],
    extra_cuda_cflags=["-lineinfo", "--expt-extended-lambda"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Initialize parameters and layers
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.gamma = nn.Parameter(torch.randn(scale_shape))  # From BatchNorm
        self.beta = nn.Parameter(torch.randn(scale_shape))   # From BatchNorm
        self.running_mean = nn.Parameter(torch.zeros(scale_shape), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(scale_shape), requires_grad=False)
        self.eps = eps
        self.momentum = momentum

        # Initialize fused kernel
        self.fused_forward = fused_forward

    def forward(self, x):
        # Execute the fused kernel
        return self.fused_forward.fused_forward_cuda(
            x,
            self.weight,
            self.bias,
            self.scale,
            self.running_mean,
            self.running_var,
            self.gamma,
            self.beta,
            self.eps,
            self.momentum
        )

def get_inputs():
    batch_size = 16384
    in_features = 4096
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    in_features = 4096
    out_features = 4096
    scale_shape = (out_features,)
    return [in_features, out_features, scale_shape]