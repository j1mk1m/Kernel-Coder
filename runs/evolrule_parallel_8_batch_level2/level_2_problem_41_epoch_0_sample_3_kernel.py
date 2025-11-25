import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define fused GEMM + BatchNorm + GELU + ReLU kernel
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Forward declarations for kernel functions
__global__ void fused_gemm_bnorm_gelu_relu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_running_mean,
    const float* __restrict__ bn_running_var,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    float eps) {

    // Calculate grid and thread indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    // Compute GEMM
    int row = idx / out_features;
    int col = idx % out_features;
    float sum = 0.0;
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum += bias[col];

    // BatchNorm computation
    float x_hat = (sum - bn_running_mean[col]) / sqrt(bn_running_var[col] + eps);
    float bn_out = bn_weight[col] * x_hat + bn_bias[col];

    // Apply GELU and ReLU
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    float gelu = 0.5 * bn_out * (1 + tanh(sqrt(2.0 / M_PI) * (bn_out + 0.044715 * pow(bn_out, 3))));
    float relu = fmaxf(gelu, 0.0f);

    output[idx] = relu;
}

torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    float eps) {

    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    const int threads = 256;
    const dim3 blocks((batch_size * out_features + threads - 1) / threads, 1, 1);

    fused_gemm_bnorm_gelu_relu<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        eps
    );

    return output;
}
"""

cpp_header = """
#include <torch/extension.h>
torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    float eps);
"""

fused_op = load_inline(
    name="fused_gemm_bnorm_gelu_relu",
    cpp_sources=cpp_header,
    cuda_sources=fused_kernel_source,
    functions=["fused_forward"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_cuda_cflags=["-arch=sm_86"]
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.bn_weight = nn.Parameter(torch.empty(out_features))
        self.bn_bias = nn.Parameter(torch.empty(out_features))
        self.bn_running_mean = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.bn_running_var = nn.Parameter(torch.ones(out_features), requires_grad=False)
        
        # Initialize weights and biases similar to PyTorch's Linear and BatchNorm
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.ones_(self.bn_weight)
        nn.init.zeros_(self.bn_bias)

        # Register the fused operator
        self.fused_op = fused_op

    def forward(self, x):
        return self.fused_op.fused_forward(
            x,
            self.weight,
            self.bias,
            self.bn_weight,
            self.bn_bias,
            self.bn_running_mean,
            self.bn_running_var,
            1e-5  # Using standard PyTorch epsilon for BatchNorm
        )

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]