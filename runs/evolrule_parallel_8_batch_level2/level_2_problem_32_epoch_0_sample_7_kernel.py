import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the fused CUDA kernel for scaling and min operation
fusion_source = """
#include <torch/extension.h>

template <typename scalar_t>
__global__ void fused_scale_min_kernel(
    const scalar_t* input, scalar_t scale, scalar_t* output,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x;
    int b = idx / (H * W);
    int rem = idx % (H * W);
    int h = rem / W;
    int w = rem % W;
    int c = threadIdx.x;

    extern __shared__ scalar_t shared[];

    scalar_t val = 0.0;
    if (c < C) {
        int input_offset = (b * C + c) * H * W + h * W + w;
        val = input[input_offset] * scale;
        shared[c] = val;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        scalar_t min_val = shared[0];
        for (int i = 1; i < C; i++) {
            if (shared[i] < min_val) {
                min_val = shared[i];
            }
        }
        int output_offset = (b * H + h) * W + w;
        output[output_offset] = min_val;
    }
}

torch::Tensor fused_scale_min_cuda(torch::Tensor input, float scale_factor) {
    int B = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    auto output = torch::zeros({B, 1, H, W}, input.options());

    int num_blocks = B * H * W;
    int block_size = C;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_scale_min_cuda", ([&] {
        int shared_mem_size = C * sizeof(scalar_t);
        fused_scale_min_kernel<scalar_t>
            <<<num_blocks, block_size, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            static_cast<scalar_t>(scale_factor),
            output.data_ptr<scalar_t>(),
            B, C, H, W
        );
    }));

    return output;
}
"""

fusion_cpp_source = "torch::Tensor fused_scale_min_cuda(torch::Tensor input, float scale_factor);"

# Load the fused CUDA kernel
fusion_mod = load_inline(
    name="fused_scale_min",
    cpp_sources=fusion_cpp_source,
    cuda_sources=fusion_source,
    functions=["fused_scale_min_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.fused_scale_min = fusion_mod

    def forward(self, x):
        x = self.conv(x)
        return self.fused_scale_min.fused_scale_min_cuda(x, self.scale_factor)

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]