import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Inline CUDA kernel for reverse cumulative sum
        reverse_cumsum_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void reverse_cumsum_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            int64_t dim_size,
            int64_t outer_dim,
            int64_t inner_dim,
            int64_t dim) {

            extern __shared__ scalar_t shared_mem[];
            int batch_idx = blockIdx.x / outer_dim;
            int inner_idx = blockIdx.x % inner_dim;
            int thread_idx = threadIdx.x;

            // Compute the offset along the dimension
            int offset = blockIdx.x * dim_size + thread_idx;

            // Load data into shared memory
            scalar_t val = (thread_idx < dim_size) ?
                input[batch_idx * outer_dim * dim_size * inner_dim +
                      (dim_size - 1 - thread_idx) * inner_dim + inner_idx] : 0;

            shared_mem[thread_idx] = val;
            __syncthreads();

            // Bitwise parallel prefix sum (reverse direction)
            for (int stride = 1; stride <= dim_size; stride *= 2) {
                int index = 2 * stride - 1 - thread_idx;
                if (index < dim_size && thread_idx >= stride) {
                    shared_mem[thread_idx] += shared_mem[index];
                }
                __syncthreads();
            }

            // Write back the result in reverse order
            if (thread_idx < dim_size) {
                int global_idx = batch_idx * outer_dim * dim_size * inner_dim +
                                (dim_size - 1 - thread_idx) * inner_dim + inner_idx;
                output[global_idx] = shared_mem[thread_idx];
            }
            __syncthreads();
        }

        torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int64_t dim) {
            auto input_size = input.sizes().vec();
            auto input_dims = input.dim();
            auto dim_size = input.size(dim);

            // Compute outer and inner dimensions
            int64_t outer_dim = 1;
            for (int i = 0; i < dim; ++i) {
                outer_dim *= input.size(i);
            }
            int64_t inner_dim = 1;
            for (int i = dim + 1; i < input_dims; ++i) {
                inner_dim *= input.size(i);
            }

            auto output = torch::empty_like(input);
            dim3 blocks(outer_dim * inner_dim);
            dim3 threads(dim_size);
            size_t smem_size = threads.x * sizeof(float);

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reverse_cumsum_cuda", ([&] {
                reverse_cumsum_kernel<scalar_t><<<blocks, threads, smem_size, input.get_device()>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    dim_size,
                    outer_dim,
                    inner_dim,
                    dim);
            }));

            cudaDeviceSynchronize();
            return output;
        }
        """

        reverse_cumsum_cpp_source = """
        torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int64_t dim);
        """

        # Load the CUDA kernel
        self.reverse_cumsum = load_inline(
            name="reverse_cumsum",
            cpp_sources=reverse_cumsum_cpp_source,
            cuda_sources=reverse_cumsum_source,
            functions=["reverse_cumsum_cuda"],
            verbose=True,
        )

    def forward(self, x):
        return self.reverse_cumsum.reverse_cumsum_cuda(x, self.dim)

def get_inputs():
    batch_size = 32768
    input_shape = (32768,)
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [1]