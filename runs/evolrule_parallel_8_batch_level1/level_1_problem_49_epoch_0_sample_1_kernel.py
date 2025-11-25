import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.max_kernel = load_inline(
            name='max_reduction_cuda',
            cpp_sources=f"""
                torch::Tensor max_reduction_cuda(torch::Tensor x, int64_t dim);
            """,
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <limits>

                template <typename scalar_t>
                __global__ void max_kernel(const scalar_t* __restrict__ x,
                                           scalar_t* __restrict__ out,
                                           const int64_t dim_size,
                                           const int64_t outer_dim,
                                           const int64_t inner_dim) {{
                    int block_idx = blockIdx.x;
                    int tid = threadIdx.x;

                    extern __shared__ scalar_t smem[];
                    scalar_t* shared_data = smem;
                    scalar_t* sdata = shared_data + dim_size;

                    // Compute outer and inner indices
                    int outer = block_idx / inner_dim;
                    int inner = block_idx % inner_dim;

                    int base = outer * (dim_size * inner_dim) + inner;

                    // Load data into shared memory
                    int elements_per_thread = (dim_size + blockDim.x - 1) / blockDim.x;
                    int start = tid * elements_per_thread;
                    int end = min(start + elements_per_thread, dim_size);

                    for (int pos = start; pos < end; ++pos) {{
                        int global_idx = base + pos * inner_dim;
                        shared_data[pos] = x[global_idx];
                    }}

                    __syncthreads();

                    // Compute local max
                    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();
                    for (int pos = 0; pos < dim_size; pos += blockDim.x) {{
                        int idx = pos + tid;
                        if (idx < dim_size) {{
                            local_max = max(local_max, shared_data[idx]);
                        }}
                    }}

                    // Store to sdata
                    sdata[tid] = local_max;
                    __syncthreads();

                    // Block reduction
                    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
                        if (tid < s) {{
                            sdata[tid] = max(sdata[tid], sdata[tid + s]);
                        }}
                        __syncthreads();
                    }}

                    if (tid == 0) {{
                        out[block_idx] = sdata[0];
                    }}
                }}

                torch::Tensor max_reduction_cuda(torch::Tensor x, int64_t dim) {{
                    auto x_size = x.sizes().vec();
                    auto output_size = x.sizes().vec();
                    output_size[dim] = 1;

                    int64_t outer_dim = 1;
                    for (int i = 0; i < dim; ++i) {{
                        outer_dim *= x.size(i);
                    }}
                    int64_t inner_dim = 1;
                    for (int i = dim + 1; i < x.dim(); ++i) {{
                        inner_dim *= x.size(i);
                    }}

                    int64_t dim_size = x.size(dim);
                    int block_size = 256;
                    int grid_size = outer_dim * inner_dim;

                    auto out = torch::empty(output_size, x.options());

                    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "max_reduction_cuda", ([&] {{
                        int sm_size = (dim_size + block_size) * sizeof(scalar_t);
                        auto stream = at::cuda::getCurrentCUDAStream();
                        max_kernel<scalar_t><<<grid_size, block_size, sm_size, stream>>>(
                            x.data_ptr<scalar_t>(),
                            out.data_ptr<scalar_t>(),
                            dim_size,
                            outer_dim,
                            inner_dim
                        );
                    }}));

                    return out.view(output_size);
                }}
            """,
            functions=['max_reduction_cuda'],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_kernel.max_reduction_cuda(x, self.dim).squeeze(self.dim)