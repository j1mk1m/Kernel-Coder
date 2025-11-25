import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Load the CUDA kernel
        self.cumprod_cuda = load_inline(
            name="cumprod_cuda",
            cpp_sources="""
                torch::Tensor cumprod_cuda(torch::Tensor input, int dim);
            """,
            cuda_sources="""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <cuda_fp16.h>
                #include <ATen/cuda/CUDAContext.h>

                template <typename scalar_t>
                __global__ void block_scan_kernel(
                    scalar_t* __restrict__ out,
                    const scalar_t* __restrict__ in,
                    int rows,
                    int cols,
                    int dim
                ) {
                    extern __shared__ scalar_t shared_mem[];
                    int tid = threadIdx.x;
                    int row = blockIdx.x;

                    // Each block handles one row (since dim=1 is along columns)
                    // Each thread handles one element in the row
                    const int col = tid;

                    // Load data into shared memory
                    if (col < cols) {
                        shared_mem[tid] = in[row * cols + col];
                    } else {
                        shared_mem[tid] = 1; // Identity element for product
                    }
                    __syncthreads();

                    // Up-sweep phase (builds the tree of products)
                    for (int stride = 1; stride <= cols; stride *= 2) {
                        int index = 2 * stride * tid;
                        if (index < cols && index + stride < cols) {
                            shared_mem[index + stride] *= shared_mem[index];
                        }
                        __syncthreads();
                    }

                    // Down-sweep phase (accumulates the products)
                    for (int stride = cols / 2; stride >= 1; stride /= 2) {
                        __syncthreads();
                        int index = 2 * stride * tid;
                        if (index < cols && index + stride < cols) {
                            scalar_t temp = shared_mem[index] * shared_mem[index + stride];
                            shared_mem[index + stride] = temp;
                            shared_mem[index] *= shared_mem[index + stride];
                        }
                    }

                    // Write back to output
                    if (col < cols) {
                        out[row * cols + col] = shared_mem[col];
                    }
                }

                torch::Tensor cumprod_cuda(torch::Tensor input, int dim) {
                    const int rows = input.size(0);
                    const int cols = input.size(1);

                    auto output = torch::empty_like(input);
                    const int threads = 256; // Tuned for 32768 elements per row
                    const dim3 blocks(rows);
                    const dim3 threads_per_block(threads);

                    // Each block processes a row, using shared memory for that row
                    // Shared memory size needed is cols * sizeof(float)
                    // Since cols=32768 and 32768 * 4 bytes = 128KB, which is under the limit
                    block_scan_kernel<float><<<blocks, threads_per_block, cols * sizeof(float)>>>(
                        output.data_ptr<float>(),
                        input.data_ptr<float>(),
                        rows,
                        cols,
                        dim
                    );

                    cudaDeviceSynchronize();
                    return output;
                }
            """,
            functions=["cumprod_cuda"],
            verbose=True
        )

    def forward(self, x):
        # Check if dim is 1, as the kernel is specialized for that
        assert self.dim == 1, "Custom kernel only supports dim=1"
        return self.cumprod_cuda.cumprod_cuda(x, self.dim)

# Ensure the input generation is compatible (on CUDA)
def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]