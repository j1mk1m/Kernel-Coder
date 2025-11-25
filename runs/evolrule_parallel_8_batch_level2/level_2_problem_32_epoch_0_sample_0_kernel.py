import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

        # Custom fused kernel for scaling and min operation
        fused_kernel = load_inline(
            name='fused_scale_min',
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <cuda_runtime.h>

            template <typename scalar_t>
            __global__ void fused_scale_min(
                const scalar_t* __restrict__ input,
                scalar_t* __restrict__ output,
                int B, int C, int H, int W,
                scalar_t scale) {{
                int batch = blockIdx.x;
                int h = blockIdx.y;
                int w = blockIdx.z;

                if (batch >= B || h >= H || w >= W)
                    return;

                scalar_t min_val = input[batch * C * H * W + 0 * H * W + h * W + w] * scale;
                for (int c = 1; c < C; ++c) {{
                    int idx = batch * C * H * W + c * H * W + h * W + w;
                    scalar_t val = input[idx] * scale;
                    if (val < min_val) min_val = val;
                }}
                output[batch * H * W + h * W + w] = min_val;
            }}

            at::Tensor fused_scale_min_cuda(at::Tensor input, at::Scalar scale) {{
                const int B = input.size(0);
                const int C = input.size(1);
                const int H = input.size(2);
                const int W = input.size(3);

                auto output = at::empty({{B, 1, H, W}}, input.options());

                dim3 blocks(B, H, W);
                dim3 threads(1);

                AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_scale_min", ([&] {{
                    fused_scale_min<scalar_t><<<blocks, threads>>>(
                        input.data<scalar_t>(),
                        output.data<scalar_t>(),
                        B, C, H, W,
                        scale.to<scalar_t>());
                }}));

                return output;
            }}
            """,
            functions=['fused_scale_min_cuda'],
            verbose=False
        )

        self.fused_op = fused_kernel

    def forward(self, x):
        x = self.conv(x)
        return self.fused_op.fused_scale_min_cuda(x, self.scale_factor).contiguous()

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]