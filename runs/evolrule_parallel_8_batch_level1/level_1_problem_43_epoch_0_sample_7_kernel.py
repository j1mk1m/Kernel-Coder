import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 3D Max Pooling
max_pool_3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void max_pool3d_forward(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int channel = blockIdx.w;
    int batch = blockIdx.v;

    if (batch >= batch_size || channel >= channels || 
        idx >= out_depth || idy >= out_height || idz >= out_width) {
        return;
    }

    int in_d_start = idx * stride_d - padding_d;
    int in_h_start = idy * stride_h - padding_h;
    int in_w_start = idz * stride_w - padding_w;

    scalar_t max_val = -FLT_MAX;
    int max_idx = -1;

    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int d = in_d_start + kd * dilation_d;
                int h = in_h_start + kh * dilation_h;
                int w = in_w_start + kw * dilation_w;

                if (d < 0 || d >= in_depth || h < 0 || h >= in_height || w < 0 || w >= in_width) {
                    continue;
                }

                int in_offset = ((batch * channels + channel) * in_depth + d) * in_height * in_width +
                                h * in_width + w;
                scalar_t val = input[in_offset];

                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }

    int out_offset = ((batch * channels + channel) * out_depth + idx) * out_height * out_width +
                     idy * out_width + idz;
    output[out_offset] = max_val;
}

std::vector<int64_t> compute_output_size(
    int64_t input_depth, int64_t input_height, int64_t input_width,
    int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t padding_d, int64_t padding_h, int64_t padding_w,
    int64_t dilation_d, int64_t dilation_h, int64_t dilation_w,
    bool ceil_mode) {

    auto compute_dim = [](int64_t input_size, int64_t kernel_size, int64_t stride_size, int64_t padding_size,
                          int64_t dilation_size, bool ceil) {
        int64_t kernel_effective = (kernel_size - 1) * dilation_size + 1;
        int64_t input_size_padded = input_size + 2 * padding_size;
        int64_t numerator = ceil ? input_size_padded : (input_size_padded - kernel_effective);
        return (numerator + (stride_size - 1)) / stride_size;
    };

    int64_t out_depth = compute_dim(input_depth, kernel_d, stride_d, padding_d, dilation_d, ceil_mode);
    int64_t out_height = compute_dim(input_height, kernel_h, stride_h, padding_h, dilation_h, ceil_mode);
    int64_t out_width = compute_dim(input_width, kernel_w, stride_w, padding_w, dilation_w, ceil_mode);
    return {out_depth, out_height, out_width};
}

torch::Tensor max_pool3d_forward_cuda(torch::Tensor input, 
                                     int kernel_size, 
                                     int stride,
                                     int padding,
                                     int dilation,
                                     bool ceil_mode) {
    const auto input_size = input.sizes();
    const int batch_size = input_size[0];
    const int channels = input_size[1];
    const int in_depth = input_size[2];
    const int in_height = input_size[3];
    const int in_width = input_size[4];

    // Currently only supports square kernels and same parameters in all dimensions
    const int kernel_d = kernel_size;
    const int kernel_h = kernel_size;
    const int kernel_w = kernel_size;
    const int stride_d = stride;
    const int stride_h = stride;
    const int stride_w = stride;
    const int padding_d = padding;
    const int padding_h = padding;
    const int padding_w = padding;
    const int dilation_d = dilation;
    const int dilation_h = dilation;
    const int dilation_w = dilation;

    auto output_sizes = compute_output_size(
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        ceil_mode);

    const int out_depth = output_sizes[0];
    const int out_height = output_sizes[1];
    const int out_width = output_sizes[2];

    auto output = torch::empty({batch_size, channels, out_depth, out_height, out_width}, input.options());

    const dim3 threads(16, 16, 1);
    dim3 blocks(out_depth, out_height, out_width);
    blocks.x = (out_depth + threads.x - 1) / threads.x;
    blocks.y = (out_height + threads.y - 1) / threads.y;
    blocks.z = (out_width + threads.z - 1) / threads.z;
    blocks.z = 1; // Simplify to 3D grid for now

    // Launch kernel with dimensions (depth, height, width, channel, batch)
    auto stream = at::cuda::getCurrentCUDAStream();
    max_pool3d_forward<scalar_t><<<dim3(blocks.x, blocks.y, blocks.z), threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        batch_size, channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w);

    cudaDeviceSynchronize();
    return output;
}
"""

max_pool_3d_cpp_source = (
    "at::Tensor max_pool3d_forward_cuda(at::Tensor input, "
    "int kernel_size, int stride, int padding, int dilation, bool ceil_mode);"
)

# Compile the CUDA kernel
max_pool3d = load_inline(
    name='max_pool3d',
    cuda_sources=max_pool_3d_source,
    cpp_sources=max_pool_3d_cpp_source,
    functions=['max_pool3d_forward_cuda'],
    verbose=True,
    extra_cuda_cflags=['-std=c++14', '-arch=sm_75'],
    extra_cflags=['-std=c++14']
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, 
                 dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.max_pool3d = max_pool3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool3d.max_pool3d_forward_cuda(
            x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode
        )

# Update input generation to include CUDA tensors
def get_inputs():
    x = torch.rand(batch_size, channels, dim1, dim2, dim3).cuda()
    return [x]

def get_init_inputs():
    # Return parameters in the order required for __init__
    return [kernel_size, stride, padding, dilation, False, False]