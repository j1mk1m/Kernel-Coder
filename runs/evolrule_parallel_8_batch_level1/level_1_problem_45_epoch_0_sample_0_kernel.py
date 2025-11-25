import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 2D Average Pooling
avg_pool_2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool_2d_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4> input,
    torch::PackedTensorAccessor<scalar_t,4> output,
    int kernel_size,
    int stride,
    int pad_h,
    int pad_w
) {
    // Calculate output dimensions
    int n = blockIdx.x;
    int c = blockIdx.y;
    int oh = blockIdx.z;
    int ow = threadIdx.x;

    // Compute input coordinates with padding
    int h_start = oh * stride - pad_h;
    int w_start = ow * stride - pad_w;
    int h_end = h_start + kernel_size;
    int w_end = w_start + kernel_size;

    scalar_t sum = 0.0;
    int valid_count = 0;

    // Iterate over the kernel region
    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            // Check if current (h,w) is within input bounds
            if (h >= 0 && h < input.size(2) && w >= 0 && w < input.size(3)) {
                sum += input[n][c][h][w];
                valid_count++;
            }
        }
    }

    // Compute average and assign to output
    if (valid_count > 0) {
        output[n][c][oh][ow] = sum / valid_count;
    } else {
        // Handle case where all elements are out of bounds (set to zero?)
        output[n][c][oh][ow] = 0.0;
    }
}

std::tuple<torch::Tensor> avg_pool_2d_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // Compute output dimensions with padding
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

    // Create output tensor
    auto output = torch::zeros({batch_size, channels, output_height, output_width}, input.options());

    // Determine grid and block dimensions
    dim3 blocks(batch_size, channels, output_height);
    dim3 threads(output_width);

    // Launch kernel with template dispatch
    AT_DISPATCH_FLOATING_TYPES(input.type(), "avg_pool_2d_forward", ([&] {
        avg_pool_2d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,4>(),
            output.packed_accessor<scalar_t,4>(),
            kernel_size,
            stride,
            padding,
            padding  // Assuming same padding for height and width
        );
    }));

    return output;
}
"""

avg_pool_2d_cpp_source = (
    "std::tuple<torch::Tensor> avg_pool_2d_forward(torch::Tensor input, int kernel_size, int stride, int padding);"
)

# Compile the inline CUDA code
avg_pool_2d = load_inline(
    name="avg_pool_2d",
    cpp_sources=avg_pool_2d_cpp_source,
    cuda_sources=avg_pool_2d_source,
    functions=["avg_pool_2d_forward"],
    verbose=True,
    extra_cflags=["-D_GLIBCXX_USE_CXX11_ABI=0"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.forward_func = avg_pool_2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_func.avg_pool_2d_forward(
            x.cuda(),
            self.kernel_size,
            self.stride,
            self.padding
        )[0]

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size]