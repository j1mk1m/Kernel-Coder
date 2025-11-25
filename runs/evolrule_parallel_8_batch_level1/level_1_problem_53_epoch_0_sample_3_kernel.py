import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Compile the custom CUDA kernel
        self.min_reduction_cuda = load_inline(
            name="min_reduction",
            cpp_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <vector>
                #include <ATen/ATen.h>

                torch::Tensor min_reduction_cuda(torch::Tensor input, int64_t dim);
            """,
            cuda_sources=f"""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                #include <vector>
                #include <ATen/ATen.h>

                template <typename scalar_t>
                __global__ void min_reduction_kernel(
                    const scalar_t* __restrict__ data,
                    scalar_t* __restrict__ output,
                    int64_t dim_size,
                    int64_t other_size,
                    int64_t batch_size,
                    int64_t dim) {{
                    int batch_idx = blockIdx.x;
                    int other_idx = blockIdx.y;
                    int idx = threadIdx.x;

                    // Compute the global index based on the non-reduction dimensions
                    int output_offset = batch_idx * other_size + other_idx;
                    int input_offset = batch_idx * dim_size * other_size + other_idx * dim_size + idx;

                    // Each thread loads a value and performs a reduction
                    scalar_t min_val = std::numeric_limits<scalar_t>::max();

                    if (idx < dim_size) {{
                        min_val = data[input_offset];
                    }}

                    // Use warp-level reduction
                    for (int stride = 1; stride < dim_size; stride *= 2) {{
                        __syncthreads();
                        int peer = idx - stride;
                        if (peer >= 0 && data[input_offset] < min_val) {{
                            min_val = data[input_offset];
                        }}
                        __syncthreads();
                    }}

                    if (idx == 0) {{
                        output[output_offset] = min_val;
                    }}
                }}

                torch::Tensor min_reduction_cuda(torch::Tensor input, int64_t dim) {{
                    auto input_size = input.sizes();
                    int64_t batch_size = input_size[0];
                    int64_t dim_size = input_size[dim];
                    int64_t other_size = 1;
                    for (int i = 0; i < input.dim(); ++i) {{
                        if (i != dim) {{
                            other_size *= input_size[i];
                        }}
                    }}

                    auto output = torch::empty({{batch_size, other_size}}, input.options());

                    dim3 blocks(batch_size, other_size);
                    dim3 threads(dim_size);

                    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "min_reduction_cuda", ([&] {{
                        min_reduction_kernel<scalar_t><<<blocks, threads>>>(
                            input.data_ptr<scalar_t>(),
                            output.data_ptr<scalar_t>(),
                            dim_size,
                            other_size / batch_size,  // Assuming dim != 0
                            batch_size,
                            dim);
                    }}));

                    cudaDeviceSynchronize();
                    return output;
                }}
            """,
            functions=["min_reduction_cuda"],
            verbose=True,
            extra_cflags=["-DWITH_CUDA"],
            extra_cuda_cflags=["-lineinfo"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_reduction_cuda.min_reduction_cuda(x, self.dim)

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2).cuda()  # Ensure CUDA tensors for kernel compatibility
    return [x]

def get_init_inputs():
    return [1]  # Example, change to desired dimension