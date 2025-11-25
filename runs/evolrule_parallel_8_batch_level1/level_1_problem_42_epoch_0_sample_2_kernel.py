import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, kernel_size, stride, padding, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Define the CUDA kernel for max pooling
        max_pool_source = """
        #include <torch/extension.h>
        #include <cuda_runtime.h>

        __global__ void max_pool2d_forward_kernel(
            const float* input,
            float* output,
            int N, int C, int H, int W,
            int OH, int OW,
            int kernel_size, int stride, int padding, int dilation) {

            int h_out = blockIdx.x;
            int w_out = blockIdx.y;

            int tx = threadIdx.x;
            int ty = threadIdx.y;

            for (int n = tx; n < N; n += blockDim.x) {
                for (int c = ty; c < C; c += blockDim.y) {

                    int start_h = h_out * stride - padding;
                    int start_w = w_out * stride - padding;

                    float max_val = -INFINITY;

                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int h_in = start_h + ky * dilation;
                            int w_in = start_w + kx * dilation;

                            if (h_in >= 0 && h_in < H &&
                                w_in >= 0 && w_in < W) {
                                int input_idx = n * C * H * W +
                                               c * H * W +
                                               h_in * W + w_in;
                                float val = input[input_idx];
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }

                    int output_idx = n * C * OH * OW +
                                    c * OH * OW +
                                    h_out * OW + w_out;
                    output[output_idx] = max_val;
                }
            }
        }
        """

        # Compile the kernel using load_inline
        self.max_pool = load_inline(
            name="max_pool",
            cuda_sources=max_pool_source,
            extra_cuda_cflags=['-std=c++14'],
            functions=["max_pool2d_forward_kernel"],
            verbose=True
        )

    def forward(self, x):
        N, C, H, W = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation

        # Calculate output dimensions
        OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1
        OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // stride + 1

        output = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)

        # Define block and grid dimensions
        block_dim = (32, 32)  # Threads per block (X, Y)
        grid_dim = (OH, OW)   # Blocks per grid (X, Y)

        # Launch the kernel
        self.max_pool.max_pool2d_forward_kernel[grid_dim, block_dim](
            x.contiguous().data_ptr(),
            output.data_ptr(),
            N, C, H, W,
            OH, OW,
            kernel_size, stride, padding, dilation
        )

        return output

# Given parameters from the original problem
batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]