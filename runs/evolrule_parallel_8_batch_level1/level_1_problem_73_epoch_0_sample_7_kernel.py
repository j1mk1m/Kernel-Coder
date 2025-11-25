import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # Define the weight parameter with appropriate dimensions
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
                kernel_size,
                device='cuda'
            )
        )
        # Initialize weights similar to PyTorch's default
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Generate the CUDA kernel code with parameters substituted
        cuda_source = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define IN_CHANNELS {in_channels}
#define OUT_CHANNELS {out_channels}
#define KERNEL_SIZE {kernel_size}
#define STRIDE {stride}
#define PADDING {padding}
#define GROUPS {groups}
#define OC_PER_GROUP (OUT_CHANNELS / GROUPS)
#define IC_PER_GROUP (IN_CHANNELS / GROUPS)

__global__ void transposed_conv3d_kernel(
    const float* input,
    const float* kernel,
    float* output,
    int batch_size,
    int in_channels,
    int input_depth,
    int input_height,
    int input_width,
    int out_channels,
    int output_depth,
    int output_height,
    int output_width
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_depth * output_height * output_width) return;

    int b = idx / (out_channels * output_depth * output_height * output_width);
    int remainder = idx % (out_channels * output_depth * output_height * output_width);
    int oc = remainder / (output_depth * output_height * output_width);
    remainder %= (output_depth * output_height * output_width);
    int od = remainder / (output_height * output_width);
    remainder %= (output_height * output_width);
    int oh = remainder / output_width;
    int ow = remainder % output_width;

    int g = oc / OC_PER_GROUP;
    int oc_in_group = oc % OC_PER_GROUP;
    float sum = 0.0f;

    for (int kz = 0; kz < KERNEL_SIZE; ++kz) {{
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {{
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {{
                int id_input = (od - kz + PADDING) / STRIDE;
                int ih_input = (oh - ky + PADDING) / STRIDE;
                int iw_input = (ow - kx + PADDING) / STRIDE;

                if (id_input < 0 || id_input >= input_depth) continue;
                if (ih_input < 0 || ih_input >= input_height) continue;
                if (iw_input < 0 || iw_input >= input_width) continue;

                for (int ic_in_group = 0; ic_in_group < IC_PER_GROUP; ++ic_in_group) {{
                    int ic = g * IC_PER_GROUP + ic_in_group;
                    int kernel_out = g * OC_PER_GROUP + oc_in_group;
                    int kernel_in = g * IC_PER_GROUP + ic_in_group;

                    int kernel_offset = kernel_out * (IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE) +
                                        kernel_in * (KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE) +
                                        kz * (KERNEL_SIZE * KERNEL_SIZE) +
                                        ky * KERNEL_SIZE +
                                        kx;

                    float weight = kernel[kernel_offset];

                    int input_offset = b * in_channels * input_depth * input_height * input_width +
                                       ic * input_depth * input_height * input_width +
                                       id_input * input_height * input_width +
                                       ih_input * input_width +
                                       iw_input;

                    sum += input[input_offset] * weight;
                }}
            }}
        }}
    }}

    int output_offset = b * out_channels * output_depth * output_height * output_width +
                        oc * output_depth * output_height * output_width +
                        od * output_height * output_width +
                        oh * output_width +
                        ow;

    output[output_offset] = sum;
}}

torch::Tensor transposed_conv3d_cuda(torch::Tensor input, torch::Tensor kernel,
    int batch_size, int in_channels, int input_depth, int input_height, int input_width,
    int out_channels, int output_depth, int output_height, int output_width) {{
    auto output = torch::zeros({{
        batch_size, out_channels, output_depth, output_height, output_width
    }}, input.options());

    const int threads_per_block = 256;
    const int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    transposed_conv3d_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_depth,
        input_height,
        input_width,
        out_channels,
        output_depth,
        output_height,
        output_width
    );

    return output;
}}
"""

        # Compile the CUDA code with the substituted parameters
        self.module = load_inline(
            name=f"transposed_conv3d_{in_channels}_{out_channels}_{kernel_size}",
            cpp_sources="""#include <torch/extension.h>""",
            cuda_sources=cuda_source,
            functions=['transposed_conv3d_cuda'],
            verbose=True,
            extra_cflags=['-DIN_CHANNELS=%d' % in_channels,
                         '-DOUT_CHANNELS=%d' % out_channels,
                         '-DKERNEL_SIZE=%d' % kernel_size,
                         '-DSTRIDE=%d' % stride,
                         '-DPADDING=%d' % padding,
                         '-DGROUPS=%d' % groups,
                         '-DOC_PER_GROUP=(OUT_CHANNELS / GROUPS)',
                         '-DIC_PER_GROUP=(IN_CHANNELS / GROUPS)'],
            extra_cuda_cflags=['-lineinfo']
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, height, width = x.size()
        output_depth = (depth - 1) * self.stride + self.kernel_size - 2 * self.padding
        output_height = (height - 1) * self.stride + self.kernel_size - 2 * self.padding
        output_width = (width - 1) * self.stride + self.kernel_size - 2 * self.padding

        return self.module.transposed_conv3d_cuda(
            x,
            self.weight,
            batch_size,
            self.in_channels,
            depth,
            height,
            width,
            self.out_channels,
            output_depth,
            output_height,
            output_width
        )

# Test code
if __name__ == "__main__":
    batch_size = 4
    in_channels = 32
    out_channels = 32
    kernel_size = 3
    depth = 32
    height = 64
    width = 128
    stride = 2
    padding = 1
    groups = 4

    model = ModelNew(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
    x = torch.randn(batch_size, in_channels, depth, height, width, device='cuda')
    y = model(x)
    print(y.shape)