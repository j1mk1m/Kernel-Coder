import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool3d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool3d_kernel(
    const float* input,
    float* output,
    int N,
    int C,
    int D,
    int H,
    int W,
    int kernel_size,
    int stride,
    int padding,
    int out_D,
    int out_H,
    int out_W
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * C * out_D * out_H * out_W) return;

    int out_size = out_D * out_H * out_W;
    int nc = tid / out_size;
    int n = nc / C;
    int c = nc % C;
    int odow = tid % out_size;
    int od = odow / (out_H * out_W);
    int ohw = odow % (out_H * out_W);
    int oh = ohw / out_W;
    int ow = ohw % out_W;

    int start_d = od * stride;
    int start_h = oh * stride;
    int start_w = ow * stride;

    float sum = 0.0f;

    for (int kd = 0; kd < kernel_size; ++kd) {
        int pd = start_d + kd;
        int original_d = pd - padding;
        if (original_d < 0 || original_d >= D) continue;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ph = start_h + kh;
            int original_h = ph - padding;
            if (original_h < 0 || original_h >= H) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int pw = start_w + kw;
                int original_w = pw - padding;
                if (original_w < 0 || original_w >= W) continue;

                int input_index = 
                    n * C * D * H * W +
                    c * D * H * W +
                    original_d * H * W +
                    original_h * W +
                    original_w;
                float val = input[input_index];
                sum += val;
            }
        }
    }

    sum /= (float)(kernel_size * kernel_size * kernel_size);

    int output_index = 
        n * C * out_D * out_H * out_W +
        c * out_D * out_H * out_W +
        od * out_H * out_W +
        oh * out_W +
        ow;
    output[output_index] = sum;
}

torch::Tensor avg_pool3d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    int out_D = (D + 2 * padding - kernel_size) / stride + 1;
    int out_H = (H + 2 * padding - kernel_size) / stride + 1;
    int out_W = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({N, C, out_D, out_H, out_W}, input.options());

    int total_threads = N * C * out_D * out_H * out_W;
    const int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    avg_pool3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        kernel_size, stride, padding,
        out_D, out_H, out_W
    );

    return output;
}
"""

avg_pool3d_header = """
torch::Tensor avg_pool3d_cuda(torch::Tensor input, int kernel_size, int stride, int padding);
"""

avg_pool3d = load_inline(
    name="avg_pool3d",
    cpp_sources=avg_pool3d_header,
    cuda_sources=avg_pool3d_source,
    functions=["avg_pool3d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_pool3d.avg_pool3d_cuda(x, self.kernel_size, self.stride, self.padding)