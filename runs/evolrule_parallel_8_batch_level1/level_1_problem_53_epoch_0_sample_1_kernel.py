import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.min_reduction = load_inline(
            name="min_reduction",
            cpp_sources=f"""
                torch::Tensor min_reduction_cuda(torch::Tensor input, int64_t dim);
            """,
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <limits>

                template <typename scalar_t>
                __device__ void warp_reduce_min(volatile scalar_t* sdata, int32_ttid) {
                    sdata[tid] = (sdata[tid] < sdata[tid + 32]) ? sdata[tid] : sdata[tid + 32];
                    sdata[tid] = (sdata[tid] < sdata[tid + 16]) ? sdata[tid] : sdata[tid + 16];
                    sdata[tid] = (sdata[tid] < sdata[tid + 8]) ? sdata[tid] : sdata[tid + 8];
                    sdata[tid] = (sdata[tid] < sdata[tid + 4]) ? sdata[tid] : sdata[tid + 4];
                    sdata[tid] = (sdata[tid] < sdata[tid + 2]) ? sdata[tid] : sdata[tid + 2];
                    sdata[tid] = (sdata[tid] < sdata[tid + 1]) ? sdata[tid] : sdata[tid + 1];
                }

                template <typename scalar_t>
                __global__ void min_reduction_kernel(
                    const scalar_t* __restrict__ input,
                    scalar_t* output,
                    int64_t dim_size,
                    int64_t outer_dim,
                    int64_t inner_dim,
                    int64_t dim
                ) {{
                    extern __shared__ scalar_t shared_mem[];
                    const int tid = threadIdx.x;
                    const int block_idx = blockIdx.x;
                    const int outer = block_idx / inner_dim;
                    const int inner = block_idx % inner_dim;

                    scalar_t local_min = std::numeric_limits<scalar_t>::max();

                    // Calculate the starting index in the input tensor
                    int input_offset = outer * dim_size * inner_dim + inner;

                    for (int i = tid; i < dim_size; i += blockDim.x) {{
                        scalar_t val = input[input_offset + i * inner_dim];
                        if (val < local_min) {{
                            local_min = val;
                        }}
                    }}

                    // Write to shared memory
                    shared_mem[tid] = local_min;
                    __syncthreads();

                    // Perform block-wide reduction
                    if (blockDim.x >= 512) {{
                        if (tid < 256) {{
                            shared_mem[tid] = (shared_mem[tid] < shared_mem[tid + 256]) ? shared_mem[tid] : shared_mem[tid + 256];
                        }}
                        __syncthreads();
                    }}
                    if (blockDim.x >= 256) {{
                        if (tid < 128) {{
                            shared_mem[tid] = (shared_mem[tid] < shared_mem[tid + 128]) ? shared_mem[tid] : shared_mem[tid + 128];
                        }}
                        __syncthreads();
                    }}
                    if (blockDim.x >= 128) {{
                        if (tid < 64) {{
                            shared_mem[tid] = (shared_mem[tid] < shared_mem[tid + 64]) ? shared_mem[tid] : shared_mem[tid + 64];
                        }}
                        __syncthreads();
                    }}
                    if (tid < 32) {{
                        warp_reduce_min<scalar_t>(shared_mem, tid);
                    }}

                    __syncthreads();

                    if (tid == 0) {{
                        output[block_idx * blockDim.x + 0] = shared_mem[0];
                    }}
                }}

                torch::Tensor min_reduction_cuda(torch::Tensor input, int64_t dim) {{
                    const int64_t ndim = input.dim();
                    auto input_size = input.sizes().vec();
                    const int64_t dim_size = input.size(dim);

                    // Compute the size of the output tensor
                    auto output_size = input.sizes().vec();
                    output_size.erase(output_size.begin() + dim);
                    torch::Tensor output = torch::empty(output_size, input.options());

                    // Calculate the dimensions needed for kernel launch
                    const int64_t outer_dim = 1;
                    for (int i = 0; i < dim; ++i) {{
                        outer_dim *= input_size[i];
                    }}
                    const int64_t inner_dim = 1;
                    for (int i = dim + 1; i < ndim; ++i) {{
                        inner_dim *= input_size[i];
                    }}

                    const dim3 block(256);
                    const dim3 grid(outer_dim * inner_dim);

                    const size_t shared_mem_size = block.x * sizeof(float);

                    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "min_reduction_cuda", ([&] {{
                        min_reduction_kernel<scalar_t><<<grid, block, shared_mem_size, 
                            at::cuda::getCurrentCUDAStream()>>>(
                            input.data<scalar_t>(),
                            output.data<scalar_t>(),
                            dim_size,
                            outer_dim,
                            inner_dim,
                            dim
                        );
                    }}));

                    cudaDeviceSynchronize();
                    return output;
                }}
            """,
            functions=["min_reduction_cuda"],
            verbose=True,
            extra_cflags=["-DUSE_DEPRECATED_SHARDED"],
            extra_cuda_cflags=["-arch=sm_75"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction.min_reduction_cuda(x, self.dim)

# Ensure the input generation matches the original's requirements
def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]  # Example, change to desired dimension