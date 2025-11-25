import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matmul_kernel(float *C, float *A, float *B, int N) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int Row = blockRow * TILE_WIDTH + threadIdx.y;
    int Col = blockCol * TILE_WIDTH + threadIdx.x;

    float Cvalue = 0.0;

    for (int m = 0; m < (N + TILE_WIDTH - 1)/TILE_WIDTH; m++) {
        __shared__ float As[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

        int aRow = Row;
        int aCol = m * TILE_WIDTH + threadIdx.x;
        int bRow = m * TILE_WIDTH + threadIdx.y;
        int bCol = Col;

        if (aRow < N && aCol < N) {
            As[threadIdx.y][threadIdx.x] = A[aRow * N + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (bRow < N && bCol < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (Row < N && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1)/TILE_WIDTH, (N + TILE_WIDTH - 1)/TILE_WIDTH);

    matmul_kernel<<<blocks, threads>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), N);

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul = load_inline(
    name="matmul",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul

    def forward(self, A, B):
        A = A.cuda()
        B = B.cuda()
        return self.matmul.matmul_cuda(A, B)

def get_inputs():
    N = 4096
    A = torch.rand(N, N)
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.rand(N, N)
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]

def get_init_inputs():
    return []