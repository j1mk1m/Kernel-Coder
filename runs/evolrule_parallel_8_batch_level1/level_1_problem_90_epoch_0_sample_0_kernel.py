import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.cumprod_cuda = load_inline(
            name="cumprod_cuda",
            cpp_sources="""
            torch::Tensor cumprod_cuda(torch::Tensor x, int64_t dim);
            """,
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>

            template<typename scalar_t>
            __global__ void cumprod_kernel(scalar_t* out, const scalar_t* in, int64_t dim_size, int64_t outer_dim, int64_t inner_dim) {{
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= outer_dim * inner_dim) return;

                int outer = idx / inner_dim;
                int inner = idx % inner_dim;

                scalar_t product = 1;
                for (int d = 0; d <= inner; ++d) {{
                    int pos = outer * dim_size * inner_dim + d * inner_dim + inner;
                    product *= in[pos];
                }}
                out[idx] = product;
            }}

            torch::Tensor cumprod_cuda(torch::Tensor x, int64_t dim) {{
                auto dims = x.sizes().vec();
                int64_t dim_size = x.size(dim);
                int64_t outer_dim = 1;
                for (int i = 0; i < dim; ++i) {{
                    outer_dim *= dims[i];
                }}
                int64_t inner_dim = 1;
                for (int i = dim + 1; i < dims.size(); ++i) {{
                    inner_dim *= dims[i];
                }}

                auto out = torch::empty_like(x);
                int block_size = 256;
                int num_blocks = (outer_dim * inner_dim + block_size - 1) / block_size;

                AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "cumprod_cuda", ([&] {{
                    cumprod_kernel<scalar_t><<<num_blocks, block_size>>>(
                        out.data_ptr<scalar_t>(),
                        x.data_ptr<scalar_t>(),
                        dim_size,
                        outer_dim,
                        inner_dim);
                }}));

                cudaDeviceSynchronize();
                return out;
            }}
            """,
            functions=["cumprod_cuda"],
            verbose=True,
            extra_cflags=["-O3"],
            extra_ldflags=[""]
        )

    def forward(self, x):
        return self.cumprod_cuda.cumprod_cuda(x, self.dim)

def get_inputs():
    return [torch.rand(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [dim]

batch_size = 32768
input_shape = (32768,)
dim = 1