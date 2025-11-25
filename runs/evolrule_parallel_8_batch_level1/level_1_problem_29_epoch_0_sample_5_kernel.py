import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus using the online approximation formula
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math_constants.h>  // For CUDA's infinity and NaN

template <typename scalar_t>
__global__ void softplus_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        scalar_t xi = x[idx];
        scalar_t abs_x = fabs(xi);
        scalar_t exp_term = exp(-abs_x);
        if (abs_x > 20) {
            // For very large |x|, exp(-|x|) is negligible, so:
            // When x > 0: softplus = x
            // When x < 0: softplus = exp(x)
            // However, using the online formula, when x is large positive:
            // softplus = x + log(1 + exp(-x)) ≈ x + 0 = x
            // When x is large negative:
            // softplus = x + log(1 + exp(x)) → but x is negative, so exp(x) is small, so log(1+exp(x)) ≈ exp(x)
            // Thus, for x > 20, output x; for x < -20, output exp(x)
            if (xi > 20) {
                out[idx] = xi;
            } else {
                out[idx] = exp(xi);  // For xi < -20, exp(xi) is ~0, but we still compute it
            }
        } else {
            scalar_t log_term = log(1 + exp_term);
            out[idx] = xi + log_term;
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor x) {
    const int threads_per_block = 256;
    auto n = x.numel();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Ensure input is contiguous
    auto x_contig = x.contiguous();
    auto out = torch::empty({n}, x.options());

    // Compute grid size
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // Launch kernel with appropriate type
    AT_DISPATCH_FLOATING_TYPES(x.type(), "softplus_cuda", ([&] {
        using scalar_t = scalar_t;
        softplus_kernel<scalar_t><<<blocks_per_grid, threads_per_block, 0, stream>>>(
            x_contig.data<scalar_t>(), out.data<scalar_t>(), n);
    }));

    // Handle possible errors (though this is simplified)
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }

    // Reshape output to match input shape (since we flattened to 1D)
    return out.view(x.sizes());
}

"""

softplus_cpp_source = (
    "torch::Tensor softplus_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for Softplus
softplus = load_inline(
    name="softplus",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softplus = softplus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CUDA and contiguous
        x = x.cuda().contiguous()
        return self.softplus.softplus_cuda(x)

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []