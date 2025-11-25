import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Configuration for matrix dimensions
M = 16384 * 2
N = 16384 * 2
K = 32 * 2

# Custom CUDA kernel using cuBLASLt for optimized matrix multiplication
cublaslt_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

// Define the GEMM function using cuBLASLt
torch::Tensor cublaslt_gemm(torch::Tensor A, torch::Tensor B) {
    cublasLtHandle_t lt_handle;
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    void* workSpace = nullptr;
    size_t workSpaceSize = 0;

    // Get CUDA stream from PyTorch tensor
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Initialize cuBLASLt handle
    CUBLAS_CHECK(cublasLtCreate(&lt_handle));
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUDA_R_32F));

    // Set operation descriptors (defaults: no transpose)
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_N, sizeof(cublasOperation_t)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N, sizeof(cublasOperation_t)));

    // Create matrix layouts for A, B, and C
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, A.size(0), A.size(1), A.size(1)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, B.size(0), B.size(1), B.size(1)));
    auto C = torch::empty({A.size(0), B.size(1)}, torch::CUDA(kCUDA));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, C.size(0), C.size(1), C.size(1)));

    // Configure the operation and compute workspace
    CUBLAS_CHECK(cublasLtMatmulGetWorkSpaceSize(
        lt_handle, matmul_desc, Adesc, Bdesc, matmul_desc, Cdesc, matmul_desc,
        &workSpaceSize
    ));
    cudaMalloc(&workSpace, workSpaceSize);

    // Execute the GEMM operation
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(
        lt_handle, matmul_desc, &alpha,
        A.data_ptr<float>(), Adesc,
        B.data_ptr<float>(), Bdesc,
        &beta,
        C.data_ptr<float>(), Cdesc,
        C.data_ptr<float>(), Cdesc,
        workSpace, workSpaceSize, stream
    ));

    // Cleanup
    cudaFree(workSpace);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(matmul_desc);
    cublasLtDestroy(lt_handle);

    return C;
}

// Error handling macro
#define CUBLAS_CHECK(status)                                                         \\
    do {                                                                             \\
        cublasStatus_t err = status;                                                 \\
        if (err != CUBLAS_STATUS_SUCCESS) {                                          \\
            std::cerr << "CUBLAS error: " << err << " at line " << __LINE__ << std::endl; \\
            throw std::runtime_error("CUBLAS failed");                               \\
        }                                                                            \\
    } while(0)
"""

# Compile the inline CUDA code using cuBLASLt
cublaslt_gemm = load_inline(
    name="cublaslt_gemm",
    cpp_sources="",
    cuda_sources=cublaslt_gemm_source,
    functions=["cublaslt_gemm"],
    verbose=True,
    extra_cflags=["-I/usr/local/cuda/include"],
    extra_ldflags=["-lcublasLt", "-lcublas", "-lcudart"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cublaslt_gemm = cublaslt_gemm

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Move tensors to CUDA
        A = A.cuda()
        B = B.cuda()
        return self.cublaslt_gemm.cublaslt_gemm(A, B)

# Required functions for input generation (unchanged from original)
def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []