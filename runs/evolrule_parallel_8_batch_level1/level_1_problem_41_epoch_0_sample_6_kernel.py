import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

max_pool_1d_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t>
__global__ void max_pool_1d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_length,
    const int out_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    int batch_channel = blockIdx.y;
    int batch_idx = batch_channel / channels;
    int channel_idx = batch_channel % channels;

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_length) return;

    int start_padded = o * stride;
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

    for (int k = 0; k < kernel_size; ++k) {
        int pos_padded = start_padded + k * dilation;
        int real_pos = pos_padded - padding;
        scalar_t val = 0.0;

        if (real_pos >= 0 && real_pos < in_length) {
            val = input[batch_idx * channels * in_length +
                       channel_idx * in_length +
                       real_pos];
        }

        if (val > max_val) {
            max_val = val;
        }
    }

    int output_offset = batch_idx * channels * out_length +
                        channel_idx * out_length +
                        o;
    output[output_offset] = max_val;
}

torch::Tensor max_pool_1d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {

    auto input_ = input.contiguous();
    const auto batch_size = input_.size(0);
    const auto channels = input_.size(1);
    const auto in_length = input_.size(2);

    const int effective_kernel_size = (kernel_size - 1) * dilation + 1;
    const int padded_length = in_length + 2 * padding;
    const int out_length = (padded_length - effective_kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, out_length}, input_.options());

    const int threads_per_block = 256;
    const int blocks_per_channel = (out_length + threads_per_block - 1) / threads_per_block;
    dim3 threads(threads_per_block);
    dim3 blocks(blocks_per_channel, batch_size * channels);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool_1d_forward", ([&] {
        max_pool_1d_kernel<scalar_t><<<blocks, threads>>>(
            input_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            in_length,
            out_length,
            kernel_size,
            stride,
            padding,
            dilation);
    }));

    return output;
}
"""

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, 
                 dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

        self.max_pool_1d_cuda = load_inline(
            name="max_pool_1d_cuda",
            cuda_sources=max_pool_1d_cuda_source,
            functions=["max_pool_1d_forward"],
            verbose=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool_1d_cuda.max_pool_1d_forward(
            x.cuda(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation
        )

batch_size = 64
features = 192
sequence_length = 65536

kernel_size = 8
stride      = 1
padding     = 4
dilation    = 3            
return_indices = False

def get_inputs():
    x = torch.rand(batch_size, features, sequence_length).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]