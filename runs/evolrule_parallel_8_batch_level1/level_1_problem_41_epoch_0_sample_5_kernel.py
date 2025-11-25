import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

        # Define the custom CUDA kernel
        max_pool1d_source = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_pool1d_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    int batch_size,
    int num_features,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_length) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_features * output_length) return;

    int batch_idx = idx / (num_features * output_length);
    int rem = idx % (num_features * output_length);
    int feature_idx = rem / output_length;
    int out_pos = rem % output_length;

    int input_start = out_pos * stride - padding;
    scalar_t max_val = -INFINITY;
    int64_t max_idx = -1;
    const int L_padded = input_length + 2 * padding;

    for (int k = 0; k < kernel_size; ++k) {{
        int input_pos_candidate = input_start + k * dilation;

        if (input_pos_candidate < 0 || input_pos_candidate >= L_padded) {{
            continue;
        }}

        scalar_t val;
        if (input_pos_candidate < 0 || input_pos_candidate >= input_length) {{
            val = 0.0;
        }} else {{
            const int input_offset = batch_idx * num_features * input_length + feature_idx * input_length + input_pos_candidate;
            val = input[input_offset];
        }}

        if (val > max_val) {{
            max_val = val;
            max_idx = input_pos_candidate;
        }}
    }}

    const int output_offset = batch_idx * num_features * output_length + feature_idx * output_length + out_pos;
    output[output_offset] = max_val;
    if (indices) {{
        indices[output_offset] = max_idx;
    }}
}}

torch::Tensor max_pool1d_cuda_forward(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool return_indices) {{
    const auto batch_size = input.size(0);
    const auto num_features = input.size(1);
    const auto input_length = input.size(2);

    const int L_padded = input_length + 2 * padding;
    const int output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({{batch_size, num_features, output_length}}, options);
    torch::Tensor indices = return_indices ? torch::empty({{batch_size, num_features, output_length}}, torch::kInt64, options) : torch::Tensor();

    const int total_threads = batch_size * num_features * output_length;
    const int threads_per_block = 256;
    const int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool1d_forward", ([&] {{
        max_pool1d_forward_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size,
            num_features,
            input_length,
            kernel_size,
            stride,
            padding,
            dilation,
            output_length
        );
    }}));

    cudaDeviceSynchronize();
    return return_indices ? torch::stack({{output, indices}}, 0) : output;
}}
"""

        # Compile the CUDA kernel
        self.max_pool1d_cuda = load_inline(
            name="max_pool1d_cuda",
            cpp_sources=f"""
            torch::Tensor max_pool1d_cuda_forward(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool return_indices);
            """,
            cuda_sources=max_pool1d_source,
            functions=["max_pool1d_cuda_forward"],
            verbose=True,
            extra_cuda_cflags=["-lineinfo", "-O3"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.cuda()
        output = self.max_pool1d_cuda.max_pool1d_cuda_forward(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
        )
        if self.return_indices:
            return output[0], output[1]
        else:
            return output[0]

def get_inputs():
    batch_size = 64
    features = 192
    sequence_length = 65536
    x = torch.randn(batch_size, features, sequence_length).cuda()
    return [x]

def get_init_inputs():
    return [8, 1, 4, 3, False]