import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Load the custom CUDA kernel
        self.max_reduction = load_inline(
            name="max_reduction",
            cpp Sources='''
            torch::Tensor max_reduction_cuda(torch::Tensor input, int64_t dim);
            ''',
            cuda Sources=f'''
            #include <torch/extension.h>
            #include <cuda_runtime.h>
            #include <cstdint>

            template <typename scalar_t>
            __global__ void max_kernel(scalar_t* output, const scalar_t* input, 
                                      int64_t dim_size, int64_t outer_size, 
                                      int64_t inner_size, int64_t dim) {{
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= outer_size * inner_size) {{
                    return;
                }}

                int outer = idx / inner_size;
                int inner = idx % inner_size;
                scalar_t max_val = -INFINITY;
                for (int d = 0; d < dim_size; ++d) {{
                    int input_idx = outer * dim_size * inner_size + d * inner_size + inner;
                    max_val = max(max_val, input[input_idx]);
                }}
                output[outer * inner_size + inner] = max_val;
            }}

            torch::Tensor max_reduction_cuda(torch::Tensor input, int64_t dim) {{
                const auto dims = input.sizes().vec();
                int64_t ndim = input.dim();
                int64_t dim_size = dims[dim];
                int64_t outer_size = 1;
                for (int i = 0; i < dim; ++i) {{
                    outer_size *= dims[i];
                }}
                int64_t inner_size = 1;
                for (int i = dim + 1; i < ndim; ++i) {{
                    inner_size *= dims[i];
                }}

                auto output = torch::empty({{outer_size, inner_size}}, input.options());
                const int block_size = 256;
                const int num_elements = outer_size * inner_size;
                const int num_blocks = (num_elements + block_size - 1) / block_size;

                AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_reduction_cuda", ([&] {{
                    max_kernel<scalar_t><<<num_blocks, block_size>>>(
                        output.data_ptr<scalar_t>(),
                        input.data_ptr<scalar_t>(),
                        dim_size, outer_size, inner_size, dim);
                }}));

                return output.reshape(dims.erase(dims.begin() + dim));
            }}
            ''',
            functions=['max_reduction_cuda'],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_reduction.max_reduction_cuda(x, self.dim)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return []