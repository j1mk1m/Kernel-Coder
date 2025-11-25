import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline

TILE_WIDTH = 32

matmul_source = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH {TILE_WIDTH}

__global__ void matmul_kernel(
    const float* A, const float* B, float* C, int N) {{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {{
        __shared__ float As[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

        // Load A's tile into shared memory
        int aRow = by * TILE_WIDTH + ty;
        int aCol = m * TILE_WIDTH + tx;
        if (aRow < N && aCol < N) {{
            As[ty][tx] = A[aRow * N + aCol];
        }} else {{
            As[ty][tx] = 0.0f;
        }}

        // Load B's tile into shared memory
        int bRow = m * TILE_WIDTH + ty;
        int bCol = bx * TILE_WIDTH + tx;
        if (bRow < N && bCol < N) {{
            Bs[ty][tx] = B[bRow * N + bCol];
        }} else {{
            Bs[ty][tx] = 0.0f;
        }}

        __syncthreads();

        // Compute the partial sum
        for (int k = 0; k < TILE_WIDTH; ++k) {{
            Cvalue += As[ty][k] * Bs[k][tx];
        }}

        __syncthreads();
    }}

    if (Row < N && Col < N) {{
        C[Row * N + Col] = Cvalue;
    }}
}}

torch::Tensor matmul_forward_cuda(torch::Tensor A, torch::Tensor B) {{
    int N = A.size(0);
    auto C = torch::empty({{N, N}}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}}
"""

matmul_cpp_source = """
torch::Tensor matmul_forward_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=["matmul_forward_cuda"],
    verbose=True,
)

class CustomMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return matmul_cuda.matmul_forward_cuda(A.cuda(), B.cuda())

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = torch.matmul(grad_output, B.t())
        grad_B = torch.matmul(A.t(), grad_output)
        return grad_A, grad_B

def custom_matmul(A, B):
    return CustomMatmulFunction.apply(A, B)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return custom_matmul(A.cuda(), B.cuda())