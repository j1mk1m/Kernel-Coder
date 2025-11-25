import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
__global__ void fused_batch_norm_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int H, int W,
    float epsilon
) {
    int c = blockIdx.x;
    extern __shared__ float shared[];
    float* s_sum = (float*) shared;
    float* s_sum_sq = s_sum + blockDim.x;

    int tid = threadIdx.x;
    int num_elements = N * H * W;

    // Phase 1: Compute local sums
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = tid; i < num_elements; i += blockDim.x) {
        int idx = c * num_elements + i;
        float x = input[idx];
        local_sum += x;
        local_sum_sq += x * x;
    }

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Phase 2: Reduce sums
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float total_sum = s_sum[0];
    float total_sum_sq = s_sum_sq[0];

    // Compute mean, variance_plus_epsilon, inv_std
    float mean = total_sum / num_elements;
    float var = (total_sum_sq / num_elements) - (mean * mean);
    float variance_plus_epsilon = var + epsilon;
    float inv_std = 1.0f / sqrtf(variance_plus_epsilon);

    // Phase 3: Normalize and apply gamma/beta
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int idx = c * num_elements + i;
        float x = input[idx];
        float normalized = (x - mean) * inv_std;
        output[idx] = normalized * gamma[c] + beta[c];
    }
}

torch::Tensor fused_batch_norm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float epsilon
) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);

    const int block_size = 256;
    dim3 grid(C);
    dim3 block(block_size);
    size_t shared_mem_size = 2 * block_size * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();
    fused_batch_norm_forward_kernel<<<grid, block, shared_mem_size, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N, C, H, W,
        epsilon
    );

    return output;
}
"""

cpp_source = """
torch::Tensor fused_batch_norm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float epsilon);
"""

# Compile the CUDA code
fused_batch_norm = load_inline(
    name="fused_batch_norm",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_batch_norm_forward"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        # Ensure parameters are contiguous and on the same device as x
        gamma = self.gamma.contiguous().to(x.device)
        beta = self.beta.contiguous().to(x.device)
        # Call the CUDA kernel
        return fused_batch_norm.fused_batch_norm_forward(x, gamma, beta, 1e-5)