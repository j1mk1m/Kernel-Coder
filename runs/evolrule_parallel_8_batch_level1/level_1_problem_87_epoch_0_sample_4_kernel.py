import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

IN_CHANNELS = 64
OUT_CHANNELS = 128
BATCH_SIZE = 16
HEIGHT = 1024
WIDTH = 1024

kernel_source = f"""
#define BATCH_SIZE {BATCH_SIZE}
#define HEIGHT {HEIGHT}
#define WIDTH {WIDTH}
#define IN_CHANNELS {IN_CHANNELS}
#define OUT_CHANNELS {OUT_CHANNELS}

extern "C" {{
    __global__ void conv1x1_nhwc_kernel(float* input, float* weights, float* output) {{
        int h = blockIdx.x;
        int w = blockIdx.y;
        __shared__ float shared_weights[OUT_CHANNELS][IN_CHANNELS];
        __shared__ float shared_input[IN_CHANNELS][BATCH_SIZE];

        if (threadIdx.x == 0) {{
            for (int c_out = 0; c_out < OUT_CHANNELS; ++c_out) {{
                for (int c_in = 0; c_in < IN_CHANNELS; ++c_in) {{
                    int weight_offset = c_in * OUT_CHANNELS + c_out;
                    shared_weights[c_out][c_in] = weights[weight_offset];
                }}
            }}
        }}
        __syncthreads();

        for (int c_in = threadIdx.x; c_in < IN_CHANNELS; c_in += blockDim.x) {{
            for (int b = 0; b < BATCH_SIZE; ++b) {{
                int input_offset = b * HEIGHT * WIDTH * IN_CHANNELS +
                                  h * WIDTH * IN_CHANNELS +
                                  w * IN_CHANNELS +
                                  c_in;
                shared_input[c_in][b] = input[input_offset];
            }}
        }}
        __syncthreads();

        int c_out = threadIdx.x;
        if (c_out >= OUT_CHANNELS) return;

        for (int b = 0; b < BATCH_SIZE; ++b) {{
            float res = 0.0f;
            for (int c_in = 0; c_in < IN_CHANNELS; ++c_in) {{
                res += shared_input[c_in][b] * shared_weights[c_out][c_in];
            }}
            int output_offset = b * HEIGHT * WIDTH * OUT_CHANNELS +
                               h * WIDTH * OUT_CHANNELS +
                               w * OUT_CHANNELS +
                               c_out;
            output[output_offset] = res;
        }}
    }}
}}
"""

# Compile the CUDA kernel
conv1x1_nhwc = load_inline(
    name="conv1x1_nhwc",
    cpp_sources="",
    cuda_sources=kernel_source,
    functions=["conv1x1_nhwc_kernel"],
    verbose=True,
    extra_cuda_cflags=["-lineinfo", "-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize weights and bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape == (BATCH_SIZE, HEIGHT, WIDTH, IN_CHANNELS), "Input must be (16, 1024, 1024, 64)"
        
        output = torch.empty(BATCH_SIZE, HEIGHT, WIDTH, OUT_CHANNELS, dtype=x.dtype, device=x.device)
        
        # Reshape weights to (in_channels, out_channels)
        weights = self.weight.permute(1, 0, 2, 3).contiguous().view(IN_CHANNELS, OUT_CHANNELS)
        weights_flat = weights.view(-1)
        
        # Launch CUDA kernel
        threadsPerBlock = (OUT_CHANNELS, 1, 1)  # 128 threads (x-dimension)
        blocksPerGrid = (HEIGHT, WIDTH, 1)
        conv1x1_nhwc.conv1x1_nhwc_kernel(
            x.data_ptr(),
            weights_flat.data_ptr(),
            output.data_ptr(),
            grid=blocksPerGrid,
            block=threadsPerBlock,
            stream=torch.cuda.current_stream().cuda_stream
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias.view(1, 1, 1, -1)
        
        return output