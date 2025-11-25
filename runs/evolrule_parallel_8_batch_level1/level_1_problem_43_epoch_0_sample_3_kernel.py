import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

elementwise_maxpool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void max_pool3d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    int N, int C, int padded_D, int padded_H, int padded_W,
    int output_D, int output_H, int output_W,
    int kernel_size, int stride, int dilation,
    bool return_indices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int idx = tid; idx < N * C * output_D * output_H * output_W; idx += blockDim.x * gridDim.x) {
        int n = idx / (C * output_D * output_H * output_W);
        int c = (idx / (output_D * output_H * output_W)) % C;
        int od = (idx / (output_H * output_W)) % output_D;
        int oh = (idx / output_W) % output_H;
        int ow = idx % output_W;

        int d_start = od * stride;
        int h_start = oh * stride;
        int w_start = ow * stride;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        int64_t max_idx = -1;

        for (int kd = 0; kd < kernel_size; ++kd) {
            int d_in = d_start + kd * dilation;
            if (d_in < 0 || d_in >= padded_D)
                continue;
            for (int kh = 0; kh < kernel_size; ++kh) {
                int h_in = h_start + kh * dilation;
                if (h_in < 0 || h_in >= padded_H)
                    continue;
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int w_in = w_start + kw * dilation;
                    if (w_in < 0 || w_in >= padded_W)
                        continue;

                    const int input_offset = 
                        n * C * padded_D * padded_H * padded_W +
                        c * padded_D * padded_H * padded_W +
                        d_in * padded_H * padded_W +
                        h_in * padded_W +
                        w_in;
                    const scalar_t val = input[input_offset];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = d_in * padded_H * padded_W + h_in * padded_W + w_in;
                    }
                }
            }
        }

        const int output_offset = 
            n * C * output_D * output_H * output_W +
            c * output_D * output_H * output_W +
            od * output_H * output_W +
            oh * output_W +
            ow;

        output[output_offset] = max_val;
        if (return_indices) {
            indices[output_offset] = max_idx;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> max_pool3d_cuda(torch::Tensor input, 
                             int kernel_size,
                             int stride,
                             int padding,
                             int dilation,
                             bool return_indices,
                             bool ceil_mode) {

    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    // Compute output dimensions
    auto compute_output_size = [&](int input_dim) {
        int effective_kernel = dilation * (kernel_size - 1) + 1;
        int input_padded = input_dim + 2 * padding;
        int numerator = input_padded - effective_kernel;
        if (ceil_mode) {
            numerator += stride - 1;
        }
        int quotient = numerator / stride;
        return quotient + 1;
    };

    int output_D = compute_output_size(D);
    int output_H = compute_output_size(H);
    int output_W = compute_output_size(W);

    // Pad input
    auto options = input.options();
    auto padded_input = F::pad(input, {padding, padding, padding, padding, padding, padding});

    // Create output tensor
    auto output = torch::empty({N, C, output_D, output_H, output_W}, options);

    // Indices tensor
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty_like(output, torch::dtype(torch::kLong));
    }

    // Launch kernel
    dim3 block(256);
    int total_elements = N * C * output_D * output_H * output_W;
    int grid = (total_elements + block.x - 1) / block.x;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool3d_cuda", ([&] {
        max_pool3d_kernel<scalar_t><<<grid, block>>>(
            padded_input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            N, C, 
            padded_input.size(2), padded_input.size(3), padded_input.size(4),
            output_D, output_H, output_W,
            kernel_size, stride, dilation,
            return_indices
        );
    }));

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed: %s\\n", cudaGetErrorString(err));
        exit(-1);
    }

    return std::make_tuple(output, indices);
}
"""

cpp_sources = """
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
"""

# Compile the CUDA code
max_pool3d_cuda = load_inline(
    name="max_pool3d_cuda",
    cpp_sources=cpp_sources,
    cuda_sources=elementwise_maxpool3d_source,
    functions=["max_pool3d_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

        self.max_pool3d_cuda = max_pool3d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, indices = self.max_pool3d_cuda(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.return_indices, 
            self.ceil_mode
        )
        if self.return_indices:
            return output, indices
        else:
            return output

def get_inputs():
    x = torch.rand(batch_size, channels, dim1, dim2, dim3).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]