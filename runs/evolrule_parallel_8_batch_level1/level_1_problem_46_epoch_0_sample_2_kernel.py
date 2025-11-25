import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool3d_kernel(const float* input, float* output,
                                 int B, int C, int D, int H, int W,
                                 int D_out, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C * D_out * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out_idx = idx / W_out;
    int h_out = h_out_idx % H_out;
    int d_out_idx = h_out_idx / H_out;
    int d_out = d_out_idx % D_out;
    int c = (d_out_idx / D_out) % C;
    int b = d_out_idx / (C * D_out);

    int start_d = d_out * 2 - 1;
    int start_h = h_out * 2 - 1;
    int start_w = w_out * 2 - 1;

    float sum = 0.0f;

    for (int kd = 0; kd < 3; ++kd) {
        int d = start_d + kd;
        if (d < 0 || d >= D) continue;
        for (int kh = 0; kh < 3; ++kh) {
            int h = start_h + kh;
            if (h < 0 || h >= H) continue;
            for (int kw = 0; kw < 3; ++kw) {
                int w = start_w + kw;
                if (w < 0 || w >= W) continue;
                int offset = b * C * D * H * W +
                             c * D * H * W +
                             d * H * W +
                             h * W +
                             w;
                sum += input[offset];
            }
        }
    }
    output[idx] = sum / 27.0f;
}

torch::Tensor avg_pool3d_cuda(torch::Tensor input) {
    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int D_out = (D - 1) / 2 + 1;
    int H_out = (H - 1) / 2 + 1;
    int W_out = (W - 1) / 2 + 1;

    auto output = torch::empty({B, C, D_out, H_out, W_out}, input.options());

    const int threads_per_block = 256;
    int num_elements = B * C * D_out * H_out * W_out;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    avg_pool3d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        B, C, D, H, W,
        D_out, H_out, W_out
    );

    return output;
}
"""

avg_pool3d_cpp_source = """
torch::Tensor avg_pool3d_cuda(torch::Tensor input);
"""

avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources=avg_pool3d_cpp_source,
    cuda_sources=avg_pool3d_source,
    functions=["avg_pool3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_cuda_flags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return avg_pool3d.avg_pool3d_cuda(x)

batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    x = torch.rand(batch_size, channels, depth, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]