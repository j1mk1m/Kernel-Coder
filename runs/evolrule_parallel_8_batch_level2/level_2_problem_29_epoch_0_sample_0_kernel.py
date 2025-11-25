import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused linear + mish + mish kernel
fused_linear_mish_mish_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ scalar_t mish(scalar_t x) {
    return x * tanh(log1p(exp(x)));
}

template <typename scalar_t>
__global__ void fused_linear_mish_mish_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* output,
    int batch_size,
    int in_features,
    int out_features
) {
    const int batch_stride = out_features;
    const int in_stride = out_features * in_features;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= batch_size) return;

    scalar_t* out_ptr = output + tid * batch_stride;
    const scalar_t* in_ptr = input + tid * in_features;

    for (int out_idx = 0; out_idx < out_features; out_idx++) {
        scalar_t sum = bias[out_idx];
        for (int in_idx = 0; in_idx < in_features; in_idx++) {
            const int weight_idx = out_idx * in_features + in_idx;
            sum += in_ptr[in_idx] * weight[weight_idx];
        }

        // Apply mish twice
        sum = mish<scalar_t>(sum);
        sum = mish<scalar_t>(sum);

        out_ptr[out_idx] = sum;
    }
}

int fused_linear_mish_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    const int max_threads = 1024;
    if (out_features > max_threads) {
        // Handle larger out_features with more threads or blocks
        // For simplicity, this example assumes out_features <= max_threads
        // but in practice, a more complex grid-stride loop would be needed
        return -1;
    }

    // Launch kernel
    fused_linear_mish_mish_kernel<float><<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    cudaDeviceSynchronize();
    return 1;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_linear_mish_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    FusedLinearMishMishCUDA(input, weight, bias, output);

    return output;
}

void FusedLinearMishMishCUDA(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output
) {
    fused_linear_mish_mish_cuda(
        input,
        weight,
        bias,
        output
    );
}
"""

cuda_source = fused_linear_mish_mish_source

# Compile the fused CUDA operator
fused_linear_mish_mish = load_inline(
    name="fused_linear_mish_mish",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_linear_mish_mish_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.fused_op = fused_linear_mish_mish

        # Initialize weights and bias similar to PyTorch's Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused_op.fused_linear_mish_mish_cuda(x, self.weight, self.bias)