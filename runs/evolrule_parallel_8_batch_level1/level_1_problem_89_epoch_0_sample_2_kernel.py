import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.cuda_cumsum = load_inline(
            name="cuda_cumsum",
            cpp_sources="""
            torch::Tensor cuda_cumsum(torch::Tensor input, int64_t dim);
            """,
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>
            #include <stdio.h>

            template <typename T>
            __global__ void block_strided_scan_kernel(
                T* output,
                const T* input,
                int64_t size,
                int64_t elements_per_block,
                int64_t blocks_per_grid,
                int64_t dim_size,
                int64_t outer_dim,
                int64_t inner_dim) {{
                __shared__ T shared_data[1024];  // Shared memory per block

                int tid = threadIdx.x;
                int bid = blockIdx.x;

                // Calculate global index
                int idx = bid * elements_per_block + tid;

                // Load data into shared memory
                if (idx < size) {{
                    int outer = idx / dim_size;
                    int inner = idx % dim_size;
                    shared_data[tid] = input[outer * dim_size + inner];
                }} else {{
                    shared_data[tid] = 0;
                }}
                __syncthreads();

                // In-place block scan using binary-exchange method
                for (int offset = 1; offset <= tid; offset *= 2) {{
                    T dep = shared_data[tid - offset];
                    __syncthreads();
                    if (tid >= offset) shared_data[tid] += dep;
                    __syncthreads();
                }}

                // Write back to output
                if (idx < size) {{
                    int outer = idx / dim_size;
                    int inner = idx % dim_size;
                    output[idx] = shared_data[tid];
                }}

                // Handle inter-block dependencies (prefix from previous block)
                if (tid == 0 && bid > 0) {{
                    int prev_block_end = (bid * elements_per_block) - 1;
                    T prefix = output[prev_block_end];
                    for (int i = 1; i < elements_per_block; i += 1) {{
                        shared_data[i] += prefix;
                    }}
                }}
                __syncthreads();
            }}

            torch::Tensor cuda_cumsum(torch::Tensor input, int64_t dim) {{
                auto input_shape = input.sizes().vec();
                int64_t dim_size = input.size(dim);
                int64_t total_elements = input.numel();
                int64_t elements_per_block = 1024;  // Tuned for 32KB shared memory
                int64_t blocks_per_grid = (total_elements + elements_per_block - 1) / elements_per_block;

                // Compute outer and inner dimensions
                int64_t outer_dim = 1;
                for (int i = 0; i < dim; ++i) {{
                    outer_dim *= input.size(i);
                }}
                int64_t inner_dim = 1;
                for (int i = dim + 1; i < input.dim(); ++i) {{
                    inner_dim *= input.size(i);
                }}

                auto output = torch::empty_like(input);
                auto stream = at::cuda::getCurrentCUDAStream();

                // Launch kernel
                block_strided_scan_kernel<float>
                <<<blocks_per_grid, elements_per_block, 0, stream>>>(
                    output.data_ptr<float>(),
                    input.data_ptr<float>(),
                    total_elements,
                    elements_per_block,
                    blocks_per_grid,
                    dim_size,
                    outer_dim,
                    inner_dim);

                cudaDeviceSynchronize();
                return output;
            }}
            """,
            functions=["cuda_cumsum"],
            verbose=True,
        )

    def forward(self, x):
        return self.cuda_cumsum(x, self.dim)

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]