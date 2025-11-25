import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class DepthwiseConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w):
        # Compute output dimensions
        N, C, H, W = input.size()
        kernel_h, kernel_w = kernel_size_h, kernel_size_w
        H_out = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        W_out = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        output = torch.empty(N, C, H_out, W_out, dtype=input.dtype, device=input.device)

        # Kernel parameters
        TILE_H, TILE_W = 16, 16
        shmem_size = (TILE_H + kernel_h - 1) * (TILE_W + kernel_w - 1) * input.element_size()

        grid_h = (H_out + TILE_H - 1) // TILE_H
        grid_w = (W_out + TILE_W - 1) // TILE_W
        grid_size = (grid_w, grid_h, N * C)
        block_size = (TILE_W, TILE_H, 1)

        # Define the CUDA kernel
        depthwise_conv2d_forward = load_inline(
            name="depthwise_conv2d_forward",
            cuda_sources=f"""
            template <typename scalar_t>
            __global__ void depthwise_conv2d_forward_kernel(
                const scalar_t* __restrict__ input,
                const scalar_t* __restrict__ weight,
                scalar_t* __restrict__ output,
                int kernel_h, int kernel_w,
                int stride_h, int stride_w,
                int padding_h, int padding_w,
                int dilation_h, int dilation_w,
                int H_out, int W_out,
                int input_H, int input_W
            ) {{
                // Thread and block indices
                int tx = threadIdx.x;
                int ty = threadIdx.y;
                int batch = blockIdx.z / {C};
                int channel = blockIdx.z % {C};
                if (batch >= {N} || channel >= {C}) return;

                int block_h = blockIdx.y;
                int block_w = blockIdx.x;
                int h_start = block_h * {TILE_H};
                int w_start = block_w * {TILE_W};

                int h_out = h_start + ty;
                int w_out = w_start + tx;

                if (h_out >= H_out || w_out >= W_out) return;

                // Compute input region start and end
                int h_in_start = h_start * stride_h - padding_h;
                int w_in_start = w_start * stride_w - padding_w;

                // Shared memory dimensions
                int shmem_h = {TILE_H} + kernel_h - 1;
                int shmem_w = {TILE_W} + kernel_w - 1;
                __shared__ scalar_t smem[{shmem_h}*{shmem_w}];

                // Load input into shared memory
                for (int sy = ty; sy < shmem_h; sy += {TILE_H}) {{
                    for (int sx = tx; sx < shmem_w; sx += {TILE_W}) {{
                        int h_in = h_in_start + sy;
                        int w_in = w_in_start + sx;
                        int idx = sy * shmem_w + sx;
                        if (h_in >= 0 && h_in < input_H && w_in >=0 && w_in < input_W) {{
                            smem[idx] = input[batch * {C}*input_H*input_W + channel*input_H*input_W + h_in*input_W + w_in];
                        }} else {{
                            smem[idx] = 0.0f;
                        }}
                    }}
                }}
                __syncthreads();

                scalar_t sum = 0.0;
                for (int kh = 0; kh < kernel_h; ++kh) {{
                    for (int kw = 0; kw < kernel_w; ++kw) {{
                        int sy = (ty) + kh*dilation_h;
                        int sx = (tx) + kw*dilation_w;
                        if (sy < shmem_h && sx < shmem_w) {{
                            int w_idx = channel*kernel_h*kernel_w + kh*kernel_w + kw;
                            sum += smem[sy*shmem_w + sx] * weight[w_idx];
                        }}
                    }}
                }}

                int out_idx = batch*{C}*H_out*W_out + channel*H_out*W_out + h_out*W_out + w_out;
                output[out_idx] = sum;
            }}

            """,
            extra_cuda_cflags=['-std=c++14'],
            verbose=False
        )

        # Launch kernel
        kernel = depthwise_conv2d_forward.depthwise_conv2d_forward_kernel.cuda()
        kernel[grid_size, block_size, shmem_size](
            input.contiguous(),
            weight.contiguous(),
            output,
            kernel_h, kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            dilation_h, dilation_w,
            H_out, W_out,
            H, W
        )

        ctx.save_for_backward(input, weight)
        ctx.params = (kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, H, W, H_out, W_out)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # This part is omitted for brevity but would involve a backward kernel
        raise NotImplementedError("Backward pass not implemented in this example")

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h=1, stride_w=1, padding_h=0, padding_w=0, dilation_h=1, dilation_w=1, groups=1, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size_h, kernel_size_w))
        self.kernel_size = (kernel_size_h, kernel_size_w)
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)
        self.groups = groups
        self.bias = bias

    def forward(self, x):
        return DepthwiseConv2dFunction.apply(
            x, self.weight, 
            self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )

# Ensure compatibility with original code's get_inputs and get_init_inputs
def get_inputs():
    x = torch.rand(32, 128, 128, 256).cuda()
    return [x]

def get_init_inputs():
    return [32, 128, 3, 7, 1, 1, 0, 0, 1, 1]