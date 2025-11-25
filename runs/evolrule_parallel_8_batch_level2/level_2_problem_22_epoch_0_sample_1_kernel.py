import torch
import torch.nn as nn

from torch.utils.cpp_extension import load_inline

# Custom fused CUDA kernel for matmul, scaling, residual addition, clamp, logsumexp, and mish
fused_kernel_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float mish_device(float x) {
    return x * tanh( log(1 + exp(x)) );
}

template <typename scalar_t>
__global__ void fused_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* output,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const float scale_factor,
    const float clamp_min,
    const float clamp_max) {

    const int batch_idx = blockIdx.x;
    const int hidden_idx = threadIdx.x;

    __shared__ scalar_t shared_data[1024]; // Assuming hidden_size is 8192, but need to adjust based on actual size

    // Compute matmul: input * weight + bias (if any)
    scalar_t val = 0.0;
    for (int i = 0; i < input_size; ++i) {
        val += input[batch_idx * input_size + i] * weight[hidden_idx * input_size + i];
    }
    val += bias[hidden_idx]; // Assuming bias exists in Linear layer

    // Scaling and residual add (x * scale + x = x*(scale +1))
    val *= (scale_factor + 1.0); // Because original code has x = x * scale_factor followed by x += x (equivalent to *2)

    // Clamp
    if (val < clamp_min) val = clamp_min;
    if (val > clamp_max) val = clamp_max;

    // Store intermediate value for logsumexp
    shared_data[hidden_idx] = val;
    __syncthreads();

    // Compute logsumexp over dim 1 (hidden_size)
    // Each thread computes partial sum, then reduction
    scalar_t sum_exp = 0.0;
    for (int i = 0; i < hidden_size; i += blockDim.x) {
        int idx = i + threadIdx.x;
        if (idx < hidden_size) {
            sum_exp += exp(shared_data[idx]);
        }
    }
    __syncthreads();

    // Reduction step (using warp-level reduction for simplicity, assuming 256 threads per block)
    for (int stride = blockDim.x/2; stride > 0; stride >>=1) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, stride);
    }

    scalar_t log_sum_exp = log(sum_exp);
    shared_data[threadIdx.x] = log_sum_exp;
    __syncthreads();

    // Apply Mish activation
    scalar_t mish_val = shared_data[threadIdx.x]; // log_sum_exp
    mish_val = mish_device(mish_val);

    // Final output is log_sum_exp * mish(log_sum_exp)
    output[batch_idx * hidden_size + hidden_idx] = log_sum_exp * mish_val;
}

torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor,
    float clamp_min,
    float clamp_max) {

    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int hidden_size = weight.size(0); // Assuming weight is [hidden_size, input_size]

    auto output = torch::empty({batch_size, hidden_size}, input.options());

    dim3 blocks(batch_size);
    dim3 threads(hidden_size); // Adjust block size based on maximum threads per block

    // Launch kernel with appropriate grid and block dimensions
    // Note: This may need tuning based on actual dimensions and CUDA limits
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_forward", ([&] {
        fused_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            input_size,
            hidden_size,
            scale_factor,
            clamp_min,
            clamp_max);
    }));

    return output;
}
"""

# Compile the fused CUDA kernel
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources="",
    cuda_sources=fused_kernel_source,
    functions=["fused_forward"],
    verbose=True,
    extra_cuda_cflags=["-arch=sm_70"],
)

class ModelNew(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        scale_factor: float,
        clamp_min: float,
        clamp_max: float,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.fused_forward_op = fused_ops.fused_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous and on CUDA
        x = x.contiguous()
        weight = self.linear.weight.t().contiguous()  # Transpose for matmul compatibility
        bias = self.linear.bias.contiguous()

        return self.fused_forward_op(
            x,
            weight,
            bias,
            self.scale_factor,
            self.clamp_min,
            self.clamp_max,
        )

# Ensure the get_inputs and get_init_inputs remain unchanged as per requirements
def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]