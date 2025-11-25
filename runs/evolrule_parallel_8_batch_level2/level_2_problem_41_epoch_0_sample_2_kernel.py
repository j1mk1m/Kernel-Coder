import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused kernel for Gemm (Linear layer), BatchNorm, GELU, and ReLU
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void fused_gemm_batchnorm_gelu_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ bn_weight,
    const scalar_t* __restrict__ bn_bias,
    const scalar_t* __restrict__ bn_mean,
    const scalar_t* __restrict__ bn_var,
    scalar_t* output,
    int batch_size,
    int in_features,
    int out_features
) {
    CUDA_KERNEL_LOOP(index, batch_size * out_features) {
        int batch = index / out_features;
        int feat = index % out_features;

        // Gemm (Linear layer) computation
        scalar_t sum = bias[feat];
        for (int k = 0; k < in_features; ++k) {
            sum += input[batch * in_features + k] * weight[feat * in_features + k];
        }

        // BatchNorm computation
        scalar_t x_hat = (sum - bn_mean[feat]) / sqrt(bn_var[feat] + 1e-5);
        scalar_t y = bn_weight[feat] * x_hat + bn_bias[feat];

        // GELU approximation followed by ReLU
        // GELU: y = 0.5 * y * (1 + tanh(sqrt(2 / M_PI) * (y + 0.044715 * y^3)))
        // Then ReLU: max(y, 0)
        scalar_t gelu = 0.5 * y * (1.0 + tanh(sqrt(2.0 / M_PI) * (y + 0.044715 * y * y * y)));
        output[index] = fmax(gelu, static_cast<scalar_t>(0.0));
    }
}

torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int threads = 256;
    const int elements = batch_size * out_features;
    const int blocks = (elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_forward", ([&] {
        fused_gemm_batchnorm_gelu_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            weight.data<scalar_t>(),
            bias.data<scalar_t>(),
            bn_weight.data<scalar_t>(),
            bn_bias.data<scalar_t>(),
            bn_mean.data<scalar_t>(),
            bn_var.data<scalar_t>(),
            output.data<scalar_t>(),
            batch_size,
            in_features,
            out_features
        );
    }));

    cudaDeviceSynchronize();
    return output;
}
"""

fused_kernel_cpp = """
torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var
);
"""

# Compile the fused kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_kernel_cpp,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward"],
    verbose=True,
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.bn_mean = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bn_var = nn.Parameter(torch.ones(out_features), requires_grad=False)
        self.fused_forward_op = fused_ops.fused_forward

        # Initialize weights similar to Linear and BatchNorm
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.fused_forward_op(
            x,
            self.weight,
            self.bias,
            self.bn_weight,
            self.bn_bias,
            self.bn_mean,
            self.bn_var
        )

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]