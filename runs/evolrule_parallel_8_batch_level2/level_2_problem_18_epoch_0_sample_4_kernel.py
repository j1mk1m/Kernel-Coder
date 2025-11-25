import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused linear and summation kernel
fused_linear_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void fused_linear_sum_forward(
    const torch::PackedTensorAccessor<scalar_t,2> input,
    const torch::PackedTensorAccessor<scalar_t,2> weight,
    const torch::PackedTensorAccessor<scalar_t,1> bias,
    torch::PackedTensorAccessor<scalar_t,2> output) {
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= output.size(0)) return;

    scalar_t sum = 0;
    for (int i = 0; i < input.size(1); ++i) {
        for (int j = 0; j < weight.size(1); ++j) {
            sum += input[batch_idx][i] * weight[i][j];
        }
    }
    sum += bias[0];  // Assuming bias is scalar here (since sum reduces to 1 element)
    output[batch_idx][0] = sum;
}

torch::Tensor fused_linear_sum_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    const int batch_size = input.size(0);
    auto output = torch::empty({batch_size, 1}, input.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_linear_sum_forward", ([&] {
        fused_linear_sum_forward<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,2>(),
            weight.packed_accessor<scalar_t,2>(),
            bias.packed_accessor<scalar_t,1>(),
            output.packed_accessor<scalar_t,2>());
    }));

    return output;
}
"""

# Define the logsumexp kernel
logsumexp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void logsumexp_kernel(
    const torch::PackedTensorAccessor<scalar_t,2> input,
    torch::PackedTensorAccessor<scalar_t,2> output) {
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= input.size(0)) return;

    scalar_t max_val = -INFINITY;
    scalar_t sum = 0;
    for (int i = 0; i < input.size(1); ++i) {
        scalar_t val = input[batch_idx][i];
        if (val > max_val) max_val = val;
    }

    for (int i = 0; i < input.size(1); ++i) {
        scalar_t exp_val = exp(input[batch_idx][i] - max_val);
        sum += exp_val;
    }

    output[batch_idx][0] = max_val + log(sum);
}

torch::Tensor logsumexp_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    auto output = torch::empty({batch_size, 1}, input.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "logsumexp_cuda", ([&] {
        logsumexp_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,2>(),
            output.packed_accessor<scalar_t,2>());
    }));

    return output;
}
"""

# Compile the inline CUDA codes
fused_linear_sum_cpp = "torch::Tensor fused_linear_sum_forward_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
fused_linear_sum = load_inline(
    name="fused_linear_sum",
    cpp_sources=fused_linear_sum_cpp,
    cuda_sources=fused_linear_sum_source,
    functions=["fused_linear_sum_forward_cuda"],
    verbose=True
)

logsumexp_cpp = "torch::Tensor logsumexp_cuda(torch::Tensor input);"
logsumexp = load_inline(
    name="logsumexp",
    cpp_sources=logsumexp_cpp,
    cuda_sources=logsumexp_source,
    functions=["logsumexp_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Only keep the weight and bias, since the linear op is fused with summation
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.fused_linear_sum = fused_linear_sum
        self.logsumexp = logsumexp

        # Initialize parameters similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Fused Linear + Summation (sum over out_features)
        # The fused kernel computes matmul followed by sum over the out_features dimension
        # The output is (batch, 1)
        fused_out = self.fused_linear_sum.fused_linear_sum_forward_cuda(
            x, self.weight.t(), self.bias
        )

        # Apply LogSumExp once
        return self.logsumexp.logsumexp_cuda(fused_out)

# Note: The get_init_inputs and get_inputs functions remain unchanged as per original code,
# but since they are not part of the model, they are not included here.