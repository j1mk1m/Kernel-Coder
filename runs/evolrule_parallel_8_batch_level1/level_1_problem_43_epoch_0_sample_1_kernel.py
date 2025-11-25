import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D Max Pooling
maxpool3d_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void maxpool3d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits> output,
    int batch_size, int channels, int input_dim1, int input_dim2, int input_dim3,
    int kernel_size, int stride, int padding, int dilation,
    int output_dim1, int output_dim2, int output_dim3
) {
    // Calculate output indices
    int n = blockIdx.x;
    int c = blockIdx.y;
    int out_d = threadIdx.x;
    int out_h = blockIdx.z % output_dim2;
    int out_w = blockIdx.z / output_dim2;

    // Compute the start and end of the input region
    int in_d_start = -padding + dilation * (out_d * stride);
    int in_h_start = -padding + dilation * (out_h * stride);
    int in_w_start = -padding + dilation * (out_w * stride);

    int in_d_end = in_d_start + kernel_size * dilation;
    int in_h_end = in_h_start + kernel_size * dilation;
    int in_w_end = in_w_start + kernel_size * dilation;

    // Clamp to input dimensions
    in_d_start = max(in_d_start, 0);
    in_h_start = max(in_h_start, 0);
    in_w_start = max(in_w_start, 0);

    in_d_end = min(in_d_end, input_dim1);
    in_h_end = min(in_h_end, input_dim2);
    in_w_end = min(in_w_end, input_dim3);

    // Iterate over the kernel region
    scalar_t max_val = -INFINITY;
    for (int d = in_d_start; d < in_d_end; d += dilation) {
        for (int h = in_h_start; h < in_h_end; h += dilation) {
            for (int w = in_w_start; w < in_w_end; w += dilation) {
                scalar_t val = input[n][c][d][h][w];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    output[n][c][out_d][out_h][out_w] = max_val;
}

std::tuple<torch::Tensor> maxpool3d_forward(
    torch::Tensor input,
    int kernel_size, int stride, int padding, int dilation,
    bool ceil_mode, bool return_indices
) {
    // Get input dimensions
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_dim1 = input.size(2);
    int input_dim2 = input.size(3);
    int input_dim3 = input.size(4);

    // Compute output dimensions
    int output_dim1 = (input_dim1 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_dim2 = (input_dim2 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_dim3 = (input_dim3 + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    torch::Tensor output = torch::empty({batch_size, channels, output_dim1, output_dim2, output_dim3}, input.options());

    // Calculate grid and block dimensions
    dim3 threads(1); // Only handle one dimension per thread (d)
    dim3 blocks(
        batch_size,
        channels,
        output_dim2 * output_dim3
    );

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool3d_forward", ([&] {
        maxpool3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits>(),
            batch_size, channels, input_dim1, input_dim2, input_dim3,
            kernel_size, stride, padding, dilation,
            output_dim1, output_dim2, output_dim3
        );
    }));

    cudaDeviceSynchronize();
    return std::make_tuple(output);
}
"""

maxpool3d_cpp_source = """
#include <torch/extension.h>

std::tuple<torch::Tensor> maxpool3d_forward(
    torch::Tensor input,
    int kernel_size, int stride, int padding, int dilation,
    bool ceil_mode, bool return_indices
);
"""

# Compile the CUDA code
maxpool3d = load_inline(
    name="maxpool3d",
    cpp_sources=[maxpool3d_cpp_source],
    cuda_sources=[maxpool3d_source],
    functions=["maxpool3d_forward"],
    verbose=True,
    extra_cuda_cflags=['-arch=sm_90'],
    extra_cflags=['-O3'],
    extra_ldflags=[''],
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
        self.maxpool3d = maxpool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, = self.maxpool3d.maxpool3d_forward(
            x.cuda(),
            self.kernel_size, self.stride, self.padding, self.dilation,
            self.ceil_mode, self.return_indices
        )
        return output

# Ensure inputs are on the correct device
def get_inputs():
    x = torch.rand(batch_size, channels, dim1, dim2, dim3).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]