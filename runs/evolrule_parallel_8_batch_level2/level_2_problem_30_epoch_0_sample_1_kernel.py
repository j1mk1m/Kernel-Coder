import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom fused kernel for GEMM (Linear) + GroupNorm + HardTanh
fused_kernel_source = """
#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Function to perform fused GEMM + GroupNorm + HardTanh
torch::Tensor fused_forward(const torch::Tensor& input,
                           const torch::Tensor& weight,
                           const torch::Tensor& bias,
                           int64_t num_groups,
                           float eps,
                           float min_val,
                           float max_val) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0); // weight is (out_features, in_features)

    // Step 1: GEMM (input @ weight.t() + bias)
    auto gemm_out = torch::empty({batch_size, out_features}, input.options());
    const float alpha = 1.0;
    const float beta = 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublasOperation_t transa = CUBLAS_OP_T; // weight is stored as (out_features, in_features), so transpose to (in_features, out_features)
    cublasSgemm(handle,
                CUBLAS_OP_N, transa,
                batch_size, out_features, in_features,
                &alpha,
                input.data_ptr<float>(), batch_size,
                weight.data_ptr<float>(), in_features,
                &beta,
                gemm_out.data_ptr<float>(), batch_size);
    cublasDestroy(handle);

    // Add bias
    auto bias_expanded = bias.unsqueeze(0).expand_as(gemm_out);
    gemm_out.add_(bias_expanded);

    // Step 2: GroupNorm
    // Reshape to (batch_size, num_groups, features_per_group)
    const int features_per_group = out_features / num_groups;
    auto grouped = gemm_out.view({batch_size, num_groups, features_per_group});

    // Compute mean and variance along the last dimension
    auto mean = grouped.mean(-1, true /* keepdim */);
    auto var = grouped.var(-1, false /* unbiased */, true /* keepdim */);

    // Normalize
    const float eps_val = eps;
    auto x_hat = (grouped - mean) / torch::sqrt(var + eps_val);

    // Step 3: HardTanh
    auto output = x_hat.clamp(min_val, max_val);

    // Reshape back to original shape
    return output.view_as(gemm_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_forward, "Fused forward pass");
}
"""

# Compile the fused kernel
fused_forward = load_inline(
    name="fused_forward",
    cpp_sources=[""],
    cuda_sources=fused_kernel_source,
    functions=["fused_forward"],
    verbose=True,
    extra_cflags=["-I/usr/local/cuda/include"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.num_groups = num_groups
        self.eps = 1e-5  # Default PyTorch GroupNorm epsilon
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

        # Initialize weights and bias like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # Load the fused kernel
        self.fused_forward = fused_forward

    def forward(self, x):
        return self.fused_forward(
            x,
            self.weight,
            self.bias,
            self.num_groups,
            self.eps,
            self.hardtanh_min,
            self.hardtanh_max,
        )

# Compatibility with the original setup
def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]