import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused MatMul + Swish + Scaling
matmul_swish_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template <typename T>
__global__ void fused_matmul_swish_scale_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* output,
    int batch_size,
    int in_features,
    int out_features,
    T scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_features) return;

    int row = idx / out_features;
    int col = idx % out_features;

    T sum = 0;
    for (int k = 0; k < in_features; ++k) {
        sum += input[row * in_features + k] * weight[col * in_features + k];
    }

    // Add bias
    sum += bias[col];

    // Swish activation: x * sigmoid(x)
    T sigmoid = static_cast<T>(1) / (static_cast<T>(1) + exp(-sum));
    T activated = sum * sigmoid;

    // Apply scaling
    activated *= scaling_factor;

    output[idx] = activated;
}

std::tuple<torch::Tensor, torch::Tensor> fused_matmul_swish_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Scalar scaling_factor
) {
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({batch_size, out_features}, output_options);

    const int threads = 256;
    const int blocks = (batch_size * out_features + threads - 1) / threads;

    // Select kernel based on data type
    if (input.dtype() == torch::kFloat32) {
        fused_matmul_swish_scale_kernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_features,
            out_features,
            scaling_factor.to<float>()
        );
    } else if (input.dtype() == torch::kFloat16) {
        fused_matmul_swish_scale_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            reinterpret_cast<const __half*>(bias.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            batch_size,
            in_features,
            out_features,
            scaling_factor.to<float>()  // Scaling factor remains float
        );
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    cudaDeviceSynchronize();
    return std::make_tuple(output, output); // Dummy second output for backward compatibility
}
"""

matmul_swish_scale_cpp_source = """
std::tuple<torch::Tensor, torch::Tensor> fused_matmul_swish_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Scalar scaling_factor
);
"""

matmul_swish_scale = load_inline(
    name="fused_matmul_swish_scale",
    cpp_sources=matmul_swish_scale_cpp_source,
    cuda_sources=matmul_swish_scale_source,
    functions=["fused_matmul_swish_scale_cuda"],
    verbose=True,
    extra_cflags=["-gencode=arch=compute_80,code=sm_80", "-DCUDA_HAS_FP16=1"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"],
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor

        # Initialize weights and bias like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.fused_op = matmul_swish_scale

    def forward(self, x):
        # The kernel expects weight in (out_features, in_features) format
        # nn.Linear's weight is stored as (out_features, in_features) already
        # So no transpose needed
        output, _ = self.fused_op.fused_matmul_swish_scale_cuda(
            x,
            self.weight,
            self.bias,
            self.scaling_factor
        )
        return output

# Compatibility with original initialization
def get_init_inputs():
    return [in_features, out_features, scaling_factor]