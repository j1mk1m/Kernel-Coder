import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for fused Instance Normalization
instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void instance_norm_kernel(
    const T* input, T* output, 
    const T* gamma, const T* beta,
    int N, int C, int H, int W, T eps) {

    int block_idx = blockIdx.x;
    int n = block_idx / C;
    int c = block_idx % C;

    int tid = threadIdx.x;
    int elements = H * W;
    int start = tid;
    int stride = blockDim.x;

    T local_sum = 0.0;
    T local_squares = 0.0;

    // Phase 1: Accumulate sum and squares for mean/variance
    for (int i = start; i < elements; i += stride) {
        int h = i / W;
        int w = i % W;
        int pos = n * C * H * W + c * H * W + h * W + w;
        T val = input[pos];
        local_sum += val;
        local_squares += val * val;
    }

    __shared__ T s_sum[256];
    __shared__ T s_squares[256];

    s_sum[tid] = local_sum;
    s_squares[tid] = local_squares;
    __syncthreads();

    // Reduction step for sum and squares
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_squares[tid] += s_squares[tid + s];
        }
        __syncthreads();
    }

    T total_sum = s_sum[0];
    T total_squares = s_squares[0];
    T count = H * W;
    T mean = total_sum / count;
    T variance = total_squares / count - mean * mean;
    T inv_std = 1.0 / sqrt(variance + eps);

    T gamma_val = gamma[c];
    T beta_val = beta[c];

    // Phase 2: Apply normalization
    for (int i = start; i < elements; i += stride) {
        int h = i / W;
        int w = i % W;
        int pos = n * C * H * W + c * H * W + h * W + w;
        T val = input[pos];
        output[pos] = (val - mean) * inv_std * gamma_val + beta_val;
    }
}

// Host function to launch the kernel
torch::Tensor instance_norm_cuda(
    torch::Tensor input, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    int64_t N, int64_t C, int64_t H, int64_t W, 
    float eps) {

    // Ensure input is 4D and contiguous
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (N, C, H, W)");
    input = input.contiguous();

    auto output = torch::empty_like(input);
    
    const int threads_per_block = 256;
    const dim3 blocks(N * C);  // Each block handles a (n,c) group
    const dim3 threads(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "instance_norm_cuda", ([&] {
        instance_norm_kernel<scalar_t><<<blocks, threads>>>(input.data_ptr<scalar_t>(),
                                                          output.data_ptr<scalar_t>(),
                                                          gamma.data_ptr<scalar_t>(),
                                                          beta.data_ptr<scalar_t>(),
                                                          N, C, H, W, eps);
    }));

    return output;
}
"""

instance_norm_cpp_source = """
#include <torch/extension.h>

torch::Tensor instance_norm_cuda(
    torch::Tensor input, 
    torch::Tensor gamma, 
    torch::Tensor beta, 
    int64_t N, int64_t C, int64_t H, int64_t W, 
    float eps);
"""

# Compile the CUDA extension
instance_norm = load_inline(
    name="instance_norm",
    cpp_sources=instance_norm_cpp_source,
    cuda_sources=instance_norm_source,
    functions=["instance_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-arch=compute_75"]
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        # Initialize learnable parameters gamma and beta
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.instance_norm_cuda = instance_norm  # Load the compiled CUDA module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract tensor dimensions
        N, C, H, W = x.size()
        # Launch fused CUDA kernel
        return self.instance_norm_cuda.instance_norm_cuda(
            x, self.gamma, self.beta, N, C, H, W, 1e-5
        )

def get_inputs():
    # Generate input tensor on CUDA
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [features]

# Configuration parameters from original problem
batch_size = 112
features = 64
dim1 = 512
dim2 = 512