import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_matmul = nn.Parameter(torch.randn(out_features))  # Matmul's bias
        self.bias_activation = nn.Parameter(torch.randn(bias_shape))  # Bias after Swish
        self.group_norm = nn.GroupNorm(num_groups, out_features)

        # Custom fused kernel
        self.fused_kernel = load_inline(
            name="fused_ops",
            cuda_sources=f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void fused_ops_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ weight,
    scalar_t* __restrict__ bias_matmul,
    scalar_t* __restrict__ bias_swish,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Matrix multiplication with bias
    for (int out_idx = 0; out_idx < out_features; ++out_idx) {{
        scalar_t sum = 0;
        for (int in_idx = 0; in_idx < in_features; ++in_idx) {{
            sum += input[batch_idx * in_features + in_idx] * 
                   weight[out_idx * in_features + in_idx];
        }}
        output[batch_idx * out_features + out_idx] = sum + bias_matmul[out_idx];
    }}

    // Apply Swish activation
    for (int out_idx = 0; out_idx < out_features; ++out_idx) {{
        scalar_t x = output[batch_idx * out_features + out_idx];
        scalar_t sigmoid_x = 1 / (1 + exp(-x));
        output[batch_idx * out_features + out_idx] = x * sigmoid_x;
    }}

    // Add the activation bias
    for (int out_idx = 0; out_idx < out_features; ++out_idx) {{
        output[batch_idx * out_features + out_idx] += bias_swish[out_idx];
    }}
}}

std::vector<torch::Tensor> fused_ops(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_matmul,
    torch::Tensor bias_swish
) {{
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    
    auto output = torch::empty({{batch_size, out_features}}, dtype=input.dtype(), device=input.device());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_ops", ([&] {{
        fused_ops_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias_matmul.data_ptr<scalar_t>(),
            bias_swish.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features
        );
    }}));

    return {{output}};
}}
            """,
            functions=['fused_ops'],
            verbose=True
        )

    def forward(self, x):
        # Apply fused operations
        x = self.fused_kernel.fused_ops(
            x,
            self.weight,
            self.bias_matmul,
            self.bias_activation
        )[0]

        # GroupNorm requires contiguous input
        x = self.group_norm(x.contiguous())
        return x