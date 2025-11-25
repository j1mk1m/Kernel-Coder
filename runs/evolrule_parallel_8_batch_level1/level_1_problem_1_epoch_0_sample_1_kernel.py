import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

TILE_SIZE = 32  # Choose an appropriate tile size, e.g., 16 or 32

matrixmul_source = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE {TILE_SIZE}

__global__ void matrixMulKernel(float* C, const float* A, const float* B, int N) {{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Cvalue = 0.0;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int m = 0; m < numTiles; ++m) {{
        // Load tiles into shared memory
        int aRow = by * TILE_SIZE + ty;
        int aCol = m * TILE_SIZE + tx;
        if (aRow < N && aCol < N) {{
            sA[ty][tx] = A[aRow * N + aCol];
        }} else {{
            sA[ty][tx] = 0.0f;
        }}

        int bRow = m * TILE_SIZE + ty;
        int bCol = bx * TILE_SIZE + tx;
        if (bRow < N && bCol < N) {{
            sB[ty][tx] = B[bRow * N + bCol];
        }} else {{
            sB[ty][tx] = 0.0f;
        }}

        __syncthreads();

        // Compute the products
        for (int k = 0; k < TILE_SIZE; ++k) {{
            Cvalue += sA[ty][k] * sB[k][tx];
        }}
    }}

    // Write the result
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    if (row < N && col < N) {{
        C[row * N + col] = Cvalue;
    }}
}}

torch::Tensor matrixMulCuda(torch::Tensor A, torch::Tensor B, int N) {{
    int size = A.size(0);
    TORCH_CHECK(A.sizes() == torch::IntArrayRef({{size, size}}));
    TORCH_CHECK(B.sizes() == torch::IntArrayRef({{size, size}}));
    TORCH_CHECK(N == size);

    auto C = torch::empty({{N, N}}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    matrixMulKernel<<<blocks, threads, 0, stream>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), N);

    return C;
}}
"""

matrixmul_cpp_source = """
torch::Tensor matrixMulCuda(torch::Tensor A, torch::Tensor B, int N);
"""

# Compile the CUDA code
matrixmul = load_inline(
    name="matrixmul",
    cpp_sources=matrixmul_cpp_source,
    cuda_sources=matrixmul_source,
    functions=["matrixMulCuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrixmul = matrixmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        N = A.size(0)  # Assuming square matrices
        return self.matrixmul.matrixMulCuda(A, B, N)