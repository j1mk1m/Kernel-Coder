import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom fused CUDA kernel for Linear + scaling + residual addition
fused_linear_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_linear_add_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float scaling_factor,
    int batch_size,
    int in_features,
    int out_features
) {
    int batch_idx = blockIdx.x;
    int out_idx = threadIdx.x;

    __shared__ scalar_t shared_sum[1024]; // adjust size based on out_features
    if (out_idx < out_features) {
        scalar_t sum = 0;
        for (int i = 0; i < in_features; i++) {
            sum += input[batch_idx * in_features + i] * weight[out_idx * in_features + i];
        }
        if (bias) sum += bias[out_idx];
        sum *= scaling_factor;
        sum += input[batch_idx * in_features + out_idx]; // residual? Wait, original code has residual addition with original_x which is pre-scaling
        // Wait, correction: original code's residual is x (after matmul) + original_x (clone before scaling). Wait no:

        // Original code:
        // x = matmul(x)  # shape (batch, out_features)
        // original_x = x.clone().detach()
        // x = x * scaling_factor
        // x = x + original_x
        // So the residual is x_scaled + original_x (original_x is x before scaling?)

        // Wait, in the code, after matmul, original_x is a copy of x before scaling. Then scaling is applied, then add original_x (so the residual is scaled_x + original_x)
        // So the formula is (matmul_result * scaling) + matmul_result = matmul_result*(scaling + 1)
        // Wait that's a simplification! 

        // Wait, the original code does: x = matmul(x) --> x is the matmul result
        // original_x = x.clone()
        // x = x * scaling
        // x += original_x --> x becomes x*(scaling) + original_x (which is same as x*(scaling + 1))
        // So the entire operation can be rewritten as (matmul_result) * (scaling + 1)

        // Wait that's a crucial simplification! So the entire computation is matmul(x) * (scaling_factor + 1)
        // Because original_x is x before scaling, so x after scaling is scaling*x, then add original_x (x) gives scaling*x + x = x*(scaling + 1)
        // So the residual addition can be eliminated by just scaling the matmul result by (scaling_factor + 1)

        // Therefore, the fused kernel can compute matmul and apply (scaling_factor + 1) directly, eliminating the need for residual add
        // So the code can be optimized to:
        // sum = (matmul_result + bias) * (scaling_factor + 1)

        // This is a major optimization! So the residual addition is redundant and can be merged into the scaling step.

        // Therefore, the kernel can compute:
        sum = (sum + (bias ? bias[out_idx] : 0)) * (scaling_factor + 1);
        output[batch_idx * out_features + out_idx] = sum;
    }
}

torch::Tensor fused_linear_add_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0); // assuming weight is out x in

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int threads = out_features; // assuming out_features <= 1024
    const dim3 blocks(batch_size);
    const dim3 threadsPerBlock(threads);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_linear_add_cuda", ([&] {
        fused_linear_add_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            scaling_factor,
            batch_size,
            in_features,
            out_features
        );
    }));

    return output;
}
"""

fused_linear_add_cpp_source = """
torch::Tensor fused_linear_add_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor
);
"""

# Compile the inline CUDA code for the fused kernel
fused_linear_add = load_inline(
    name="fused_linear_add",
    cpp_sources=fused_linear_add_cpp_source,
    cuda_sources=fused_linear_kernel_source,
    functions=["fused_linear_add_cuda"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
    extra_cuda_cflags=["-std=c++14"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor
        # Initialize weights and bias like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.fused_linear_add = fused_linear_add

    def forward(self, x):
        # The fused kernel handles the matmul, bias, scaling, and residual addition
        return self.fused_linear_add.fused_linear_add_cuda(
            x,
            self.weight,
            self.bias,
            self.scaling_factor
        )