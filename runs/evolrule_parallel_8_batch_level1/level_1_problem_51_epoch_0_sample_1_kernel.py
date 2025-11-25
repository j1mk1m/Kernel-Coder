import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Define the custom CUDA kernel for argmax
        argmax_source = f"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <limits>

        template <typename scalar_t>
        __global__ void argmax_kernel(const scalar_t* input, int* output,
            int dim, int outer_dim, int inner_dim) {{
            int batch_idx = blockIdx.x / inner_dim;
            int inner_idx = blockIdx.x % inner_dim;
            int idx = batch_idx * outer_dim * inner_dim + threadIdx.x * inner_dim + inner_idx;
            int output_offset = batch_idx * inner_dim + inner_idx;

            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
            int max_idx = 0;

            for (int i = 0; i < outer_dim; ++i) {{
                int current_idx = idx + i * inner_dim;
                if (input[current_idx] > max_val) {{
                    max_val = input[current_idx];
                    max_idx = i;
                }}
            }}

            output[output_offset] = max_idx;
        }}

        torch::Tensor argmax_cuda(torch::Tensor input, int dim) {{
            int64_t dims[] = input.sizes().data();
            int batch_size = dims[0];
            int outer_dim = dims[dim];
            int inner_dim = 1;

            for (int i = 1; i < input.dim(); ++i) {{
                if (i != dim) {{
                    inner_dim *= dims[i];
                }}
            }}

            auto output_size = input.sizes().vec();
            output_size.erase(output_size.begin() + dim);
            auto output = torch::empty(output_size, input.options().dtype(torch::kInt32));

            dim3 block(outer_dim);
            dim3 grid(batch_size * inner_dim);

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_cuda", ([&] {{
                argmax_kernel<scalar_t><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<int>(),
                    dim,
                    outer_dim,
                    inner_dim
                );
            }}));

            return output;
        }}
        """

        argmax_cpp_source = (
            "torch::Tensor argmax_cuda(torch::Tensor input, int dim);"
        )

        # Compile the inline CUDA code for argmax
        self.argmax = load_inline(
            name="custom_argmax",
            cpp_sources=argmax_cpp_source,
            cuda_sources=argmax_source,
            functions=["argmax_cuda"],
            verbose=False,
            with_cuda=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.argmax.argmax_cuda(x, self.dim).to(x.device)

# Ensure the original helper functions are used
def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]

def get_init_inputs():
    return [1]