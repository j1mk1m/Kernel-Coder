import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)

        # Define the custom average pooling kernel
        avg_pool_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        template <typename scalar_t>
        __global__ void avg_pool_4x4x4_kernel(const scalar_t* __restrict__ input,
                                              scalar_t* __restrict__ output,
                                              int N, int C, int D, int H, int W,
                                              int output_D, int output_H, int output_W) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N * C * output_D * output_H * output_W) return;

            int w = idx % output_W;
            int h = (idx / output_W) % output_H;
            int d = (idx / (output_W * output_H)) % output_D;
            int c = (idx / (output_W * output_H * output_D)) % C;
            int n = idx / (output_W * output_H * output_D * C);

            int in_d = d * 4;
            int in_h = h * 4;
            int in_w = w * 4;

            scalar_t sum = 0.0;
            for (int kd = 0; kd < 4; ++kd) {
                for (int kh = 0; kh < 4; ++kh) {
                    for (int kw = 0; kw < 4; ++kw) {
                        int id = in_d + kd;
                        int ih = in_h + kh;
                        int iw = in_w + kw;
                        if (id < D && ih < H && iw < W) {
                            int input_idx = n * C * D * H * W + 
                                            c * D * H * W + 
                                            id * H * W + 
                                            ih * W + iw;
                            sum += input[input_idx];
                        }
                    }
                }
            }
            output[idx] = sum / 64.0;
        }

        at::Tensor avg_pool_4x4x4_cuda(at::Tensor input) {
            at::Device device = input.device();
            at::Tensor output = at::empty(
                {input.size(0), input.size(1),
                 (input.size(2) + 3) / 4,  // assuming padding=0 and stride=4
                 (input.size(3) + 3) / 4,
                 (input.size(4) + 3) / 4},
                input.options());

            const int N = input.size(0);
            const int C = input.size(1);
            const int D = input.size(2);
            const int H = input.size(3);
            const int W = input.size(4);
            const int output_D = (D + 3) / 4;
            const int output_H = (H + 3) / 4;
            const int output_W = (W + 3) / 4;
            const int total_elements = N * C * output_D * output_H * output_W;

            const int threads = 256;
            const int blocks = (total_elements + threads - 1) / threads;

            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool_4x4x4_cuda", ([&] {
                avg_pool_4x4x4_kernel<scalar_t><<<blocks, threads>>>(
                    input.data<scalar_t>(),
                    output.data<scalar_t>(),
                    N, C, D, H, W, output_D, output_H, output_W);
            }));

            return output;
        }
        """

        avg_pool_cpp_source = """
        at::Tensor avg_pool_4x4x4_cuda(at::Tensor input);
        """

        # Compile the custom average pooling kernel
        self.avg_pool = load_inline(
            name="avg_pool",
            cpp_sources=avg_pool_cpp_source,
            cuda_sources=avg_pool_source,
            functions=["avg_pool_4x4x4_cuda"],
            verbose=True,
            extra_cflags=["-std=c++14"],
            extra_cuda_cflags=["-std=c++14"],
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.avg_pool.avg_pool_4x4x4_cuda(x)
        return x