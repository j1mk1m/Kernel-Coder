from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Gemm + Swish Exact
gemm_swish_exact_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void gemm_swish_exact_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum * (sum > 0.0f ? sum : 0.01f * sum);
    }
}

torch::Tensor gemm_swish_exact_cuda(torch::Tensor a, torch::Tensor b) {
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);

    auto c = torch::zeros({m, n}, a.options());

    const int block_size = 16;
    const int num_blocks_x = (n + block_size - 1) / block_size;
    const int num_blocks_y = (m + block_size - 1) / block_size;

    gemm_swish_exact_kernel<<<dim3(num_blocks_x, num_blocks_y), dim3(block_size, block_size)>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);

    return c;
}
"""

gemm_swish_exact_cpp_source = (
    "torch::Tensor gemm_swish_exact_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for fused Gemm + Swish Exact
gemm_swish_exact = load_inline(
    name="gemm_swish_exact",
    cpp_sources=gemm_swish_exact_cpp_source,
    cuda_sources=gemm_swish_exact_source,
    functions=["gemm_swish_exact_cuda"],
    verbose=True,
    extra_c