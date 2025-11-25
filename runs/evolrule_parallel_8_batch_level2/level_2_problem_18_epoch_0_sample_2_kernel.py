import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused linear layer + element-wise operations
# This kernel combines matmul, sum, max, mean, and two logsumexp operations into a single kernel
fused_operations_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void fused_operations_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    // Each thread processes one element in the batch
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // Initialize variables for intermediate results
    scalar_t sum_val = 0.0;
    scalar_t max_val = -INFINITY;
    scalar_t sum_exp = 0.0;
    scalar_t sum_exp_lse = 0.0;

    // Matrix multiplication and accumulation
    for (int j = 0; j < out_features; ++j) {
        scalar_t val = 0.0;
        for (int k = 0; k < in_features; ++k) {
            val += input[batch_idx * in_features + k] * weight[j * in_features + k];
        }
        val += bias[j]; // Add bias from linear layer

        // Accumulate sum
        sum_val += val;

        // Track max value (for max operation)
        if (val > max_val) {
            max_val = val;
        }
    }

    // Compute sum and max (equivalent to first two operations)
    sum_val = sum_val; // sum
    max_val = max_val; // max

    // Compute mean (sum / out_features)
    scalar_t mean_val = sum_val / out_features;

    // First logsumexp: log(sum(exp(mean_val)))
    // Since mean_val is a scalar for each batch, exp(mean_val) summed over dimension is just exp(mean_val)
    // But since it's over dimension 1 (which is size 1), it's just exp(mean_val)
    scalar_t lse1 = log(exp(mean_val)); // which simplifies to mean_val, but let's compute correctly
    lse1 = log(exp(mean_val)); // redundant but keeping as per original steps

    // Second logsumexp: same as first, so becomes log(exp(lse1)) = lse1
    scalar_t lse2 = log(exp(lse1));

    // Store final result
    output[batch_idx] = lse2;
}

torch::Tensor fused_operations_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_features,
    int out_features
) {
    const int threads_per_block = 256;
    const int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;

    auto output = torch::empty({batch_size}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_operations_cuda", ([&] {
        fused_operations_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features
        );
    }));

    return output;
}
"""

fused_operations_cpp_source = """
torch::Tensor fused_operations_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int batch_size,
    int in_features,
    int out_features
);
"""

# Compile the fused kernel
fused_ops = load_inline(
    name="fused_operations",
    cpp_sources=fused_operations_cpp_source,
    cuda_sources=fused_operations_source,
    functions=["fused_operations_cuda"],
    verbose=True,
    extra_cflags=["-g -O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.fused_ops = fused_ops
        # Initialize weights like the original Linear layer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size = x.size(0)
        in_features = self.weight.size(1)
        out_features = self.weight.size(0)
        return self.fused_ops.fused_operations_cuda(
            x, self.weight, self.bias, batch_size, in_features, out_features
        ).unsqueeze(1)

# Update get_inputs to ensure CUDA tensors
def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]

# Adjust batch_size and features to global variables as in original
batch_size = 1024
in_features  = 8192  
out_features = 8192