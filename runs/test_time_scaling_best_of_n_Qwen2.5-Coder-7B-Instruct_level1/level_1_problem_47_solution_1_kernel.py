from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for sum reduction
sum_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_reduction_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Reduce within shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void sum_reduction(float* input, float* output, int n, int num_blocks) {
    sum_reduction_kernel<<<num_blocks, blockDim.x, sizeof(float) * blockDim.x>>>(input, output, n);
}
"""

sum_reduction_cpp_source = (
    "void sum_reduction(float* input, float* output, int n, int num_blocks);"
)

# Compile the inline CUDA code for sum reduction
sum_reduction = load_inline(
    name="sum_reduction",
    cpp_sources=sum_reduction_cpp_source,
    cuda_sources=sum_reduction_source,
    functions=["sum_reduction"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

def sum_reduction_cuda(input, n):
    num_blocks = (n + blockDim.x - 1) // blockDim.x
    output = torch.zeros(num_blocks, dtype=input.dtype, device=input.device)
    sum_reduction(input.data_ptr(), output.data_ptr(), n, num_blocks)
    return output.sum().item()

print(sum_reduction_cuda(torch.tensor([1.0, 2.0, 3.0, 4.0]), 4))