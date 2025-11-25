import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cuda_module = self._load_cuda_module()

    def _load_cuda_module(self):
        TS = 16
        kernel_code = f"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void matrixMulKernel(float* C, const float* A, const float* B, int M, int N, int K) {{
            __shared__ float tileA[{TS}][{TS}];
            __shared__ float tileB[{TS}][{TS}];

            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockIdx.x;
            int by = blockIdx.y;

            int row = by * {TS} + ty;
            int col = bx * {TS} + tx;

            if (row >= M || col >= N)
                return;

            float Cvalue = 0.0f;

            for (int m = 0; m < (K + {TS} - 1) / {TS}; m++) {{
                // Load tile of A
                int aCol = m * {TS} + tx;
                if (aCol < K) {{
                    tileA[tx][ty] = A[row * K + aCol];
                }} else {{
                    tileA[tx][ty] = 0.0f;
                }}

                // Load tile of B
                int bRow = m * {TS} + ty;
                if (bRow < K) {{
                    tileB[ty][tx] = B[bRow * N + col];
                }} else {{
                    tileB[ty][tx] = 0.0f;
                }}

                __syncthreads();

                // Compute the products for this tile
                for (int k = 0; k < {TS}; ++k) {{
                    Cvalue += tileA[k][ty] * tileB[k][tx];
                }}

                __syncthreads();
            }}

            if (row < M && col < N) {{
                C[row * N + col] = Cvalue;
            }}
        }}

        torch::Tensor matrixMulCuda(torch::Tensor A, torch::Tensor B) {{
            const int M = A.size(0);
            const int N = B.size(1);
            const int K = A.size(1);

            auto C = torch::empty({{M, N}}, A.options());

            dim3 threads({TS}, {TS});
            dim3 blocks(
                (N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y
            );

            matrixMulKernel<<<blocks, threads>>>(
                C.data_ptr<float>(),
                A.data_ptr<float>(),
                B.data_ptr<float>(),
                M, N, K
            );

            return C;
        }}
        """
        module = load_inline(
            name="matrixmul",
            cpp_sources="",
            cuda_sources=kernel_code,
            functions=["matrixMulCuda"],
            verbose=True
        )
        return module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.cuda_module.matrixMulCuda(A.cuda(), B.cuda())

# Ensure that the input dimensions are correct
M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.rand(M, K).cuda()
    B = torch.rand(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []