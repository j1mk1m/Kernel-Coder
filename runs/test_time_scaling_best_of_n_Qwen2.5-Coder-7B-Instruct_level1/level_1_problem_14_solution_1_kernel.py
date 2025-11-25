#include <torch/extension.h>
#include <cuda_runtime.h>

// Custom CUDA kernel for matrix multiplication of upper triangular matrices
__global__ void custom_matmul_upper_tri_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k <= min(row, col); ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// Custom CUDA function for matrix multiplication of upper triangular matrices
torch::Tensor custom_matmul_upper_tri_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                          (N + threads_per_block.y - 1) / threads_per_block.y);

    custom_matmul_upper_tri_kernel<<<blocks_per_grid, threads_per_block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

// Wrapper class to integrate custom CUDA function into PyTorch model
class ModelNew : public torch::nn::Module {
public:
    ModelNew() {}

    torch::Tensor forward(torch::Tensor A, torch::Tensor B) override {
        return custom_matmul_upper_tri_cuda(A, B);
    }
};