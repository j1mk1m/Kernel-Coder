import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

        # Define fused kernel for clamp + spatial softmax + scale multiplication
        self.softmax_scale_kernel = load_inline(
            name='softmax_scale',
            cuda_sources=f"""
            #include <torch/extension.h>
            #include <ATen/cuda/CUDAContext.h>
            #include <cuda.h>
            #include <cuda_runtime.h>
            #include <vector>

            template <typename scalar_t>
            __global__ void fused_clamp_softmax_scale_kernel(
                const scalar_t* input, scalar_t* output,
                const scalar_t min_val, const scalar_t max_val,
                const scalar_t* scale, int batch, int channels, int spatial_size
            ) {{
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= batch * channels * spatial_size) return;

                int b = idx / (channels * spatial_size);
                int c = (idx / spatial_size) % channels;
                int s = idx % spatial_size;

                // Clamp
                scalar_t clamped = input[idx];
                if (clamped < min_val) clamped = min_val;
                if (clamped > max_val) clamped = max_val;

                // Compute max for softmax (per channel)
                extern __shared__ scalar_t shared_max[];
                scalar_t max_val_t = clamped;
                for (int stride = spatial_size / 2; stride > 0; stride >>= 1) {{
                    __syncthreads();
                    if (s < stride) {{
                        max_val_t = max(max_val_t, shared_max[threadIdx.x + stride]);
                    }}
                    __syncthreads();
                }}
                if (threadIdx.x == 0) shared_max[0] = max_val_t;
                __syncthreads();
                scalar_t max = shared_max[0];

                // Compute exp and sum
                scalar_t exp_val = exp(clamped - max);
                scalar_t sum = 0.0;
                for (int i = 0; i < spatial_size; i += blockDim.x) {{
                    if (i + threadIdx.x < spatial_size)
                        sum += exp(clamped[b*channels*spatial_size + c*spatial_size + i]);
                }}
                __shared__ scalar_t shared_sum[1];
                sum = warpReduceSum(sum);
                if (threadIdx.x == 0) atomicAdd(shared_sum, sum);
                __syncthreads();
                scalar_t total_sum = shared_sum[0];

                // Final computation
                scalar_t softmax_val = exp_val / total_sum;
                output[idx] = softmax_val * scale[c];
            }}

            at::Tensor fused_clamp_softmax_scale(
                at::Tensor input, at::Tensor scale,
                scalar_t min_val, scalar_t max_val
            ) {{
                auto output = at::empty_like(input);
                auto batch = input.size(0);
                auto channels = input.size(1);
                auto spatial_size = input.size(2) * input.size(3) * input.size(4);
                const int threads = 256;
                const dim3 blocks((batch * channels * spatial_size + threads - 1) / threads);
                const int smem = sizeof(float) * (threads + 1);

                AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_clamp_softmax_scale", ([&] {{
                    fused_clamp_softmax_scale_kernel<scalar_t><<<blocks, threads, smem>>>(
                        input.data<scalar_t>(),
                        output.data<scalar_t>(),
                        min_val, max_val,
                        scale.data<scalar_t>(),
                        batch, channels, spatial_size
                    );
                }}));

                return output;
            }}
            """,
            functions=['fused_clamp_softmax_scale'],
            verbose=False
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        # Replace clamp, view, softmax, view, and scale multiplication with fused kernel
        scale_reshaped = self.scale.view(1, -1, 1, 1, 1)
        x = self.softmax_scale_kernel.fused_clamp_softmax_scale(
            x, scale_reshaped, self.clamp_min, self.clamp_max
        )
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]