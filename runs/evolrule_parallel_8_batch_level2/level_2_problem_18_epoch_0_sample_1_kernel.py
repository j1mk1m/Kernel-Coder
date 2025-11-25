import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for all operations
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void fused_operations_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* output,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    extern __shared__ scalar_t shared_mem[];
    scalar_t* temp = shared_mem;

    // Matrix multiplication and summation (step 1 and 2)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        scalar_t sum_val = 0.0;
        for (int i = 0; i < out_features; ++i) {
            scalar_t val = 0.0;
            for (int j = 0; j < in_features; ++j) {
                val += input[idx * in_features + j] * weight[i * in_features + j];
            }
            val += bias[i];
            sum_val += val; // Sum across out_features
        }
        // Store intermediate result in shared memory
        temp[threadIdx.x] = sum_val;
        __syncthreads();

        // Compute max (step 3)
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                temp[threadIdx.x] = fmaxf(temp[threadIdx.x], temp[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        scalar_t max_val = (threadIdx.x == 0) ? temp[0] : 0.0;
        __syncthreads();

        // Compute mean (step 4)
        scalar_t mean_val = sum_val / out_features;
        __syncthreads();

        // Compute first LogSumExp (step 5)
        scalar_t max_lse = max_val;
        scalar_t sum_exp = 0.0;
        for (int i = 0; i < out_features; ++i) {
            scalar_t exp_val = exp(temp[threadIdx.x * out_features + i] - max_lse);
            sum_exp += exp_val;
        }
        scalar_t lse = log(sum_exp) + max_lse;
        __syncthreads();

        // Compute second LogSumExp (step 6)
        max_lse = lse;
        sum_exp = exp(lse - max_lse);
        scalar_t final_lse = log(sum_exp) + max_lse;

        output[idx] = final_lse;
    }
}

torch::Tensor fused_operations_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, 1}, input.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    const int shared_size = threads * sizeof(float); // Adjust based on needed shared memory
    fused_operations_kernel<float><<<blocks, threads, shared_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_operations_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
);
"""

# Compile the fused CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=[fused_ops_cpp_source],
    cuda_sources=[fused_ops_source],
    functions=["fused_operations_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.fused_ops = fused_ops

        # Initialize weights and bias (optional, since original Model uses nn.Linear)
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.fused_ops.fused_operations_cuda(x, self.weight, self.bias)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]