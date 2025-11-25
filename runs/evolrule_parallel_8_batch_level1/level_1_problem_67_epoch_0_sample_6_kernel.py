import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# Define the im2col kernel
im2col_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void im2col_kernel(
    const float* input, float* output,
    int batch_size, int in_channels, int input_length,
    int kernel_size, int dilation, int stride,
    int output_cols, int im2col_height) {

    int b = blockIdx.z;
    if (b >= batch_size) return;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= im2col_height || col >= output_cols)
        return;

    // Compute channel and kernel_element from row
    int channel = row / kernel_size;
    int kernel_element = row % kernel_size;

    // Compute the starting position in the input
    int start = col * stride;
    int input_pos = start + kernel_element * dilation;

    // Ensure input_pos is within the padded input
    if (input_pos < 0 || input_pos >= input_length) {
        output[ b * im2col_height * output_cols + row * output_cols + col ] = 0.0f;
        return;
    }

    // Get the value from input
    int input_offset = b * in_channels * input_length + channel * input_length + input_pos;
    float val = input[input_offset];

    // Store in output
    int output_offset = b * im2col_height * output_cols + row * output_cols + col;
    output[output_offset] = val;
}

torch::Tensor im2col_cuda(
    torch::Tensor input,
    int kernel_size, int dilation, int stride) {

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2); // already padded

    int output_cols = (input_length - dilation * (kernel_size - 1)) / stride + 1;
    int im2col_height = in_channels * kernel_size;

    auto output = torch::empty({batch_size, im2col_height, output_cols}, input.options());

    dim3 block(32, 8, 1);
    int grid_x = (output_cols + block.x - 1) / block.x;
    int grid_y = (im2col_height + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y, batch_size);

    im2col_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, input_length,
        kernel_size, dilation, stride,
        output_cols, im2col_height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\\n", cudaGetErrorString(err));

    return output.view({batch_size, im2col_height, output_cols});
}
"""

im2col_cpp_source = """
torch::Tensor im2col_cuda(
    torch::Tensor input,
    int kernel_size, int dilation, int stride);
"""

im2col = load_inline(
    name="im2col_cuda",
    cpp_sources=im2col_cpp_source,
    cuda_sources=im2col_source,
    functions=["im2col_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Pad the input
        x_padded = F.pad(x, (self.padding, self.padding))

        # Split into groups
        groups = self.groups
        in_group = self.in_channels // groups
        out_group = self.out_channels // groups

        # Split input into groups
        x_split = x_padded.chunk(groups, dim=1)

        # Process each group
        outputs = []
        for g in range(groups):
            x_group = x_split[g]
            # Compute im2col for this group
            im2col_g = im2col_cuda.im2col_cuda(
                x_group,
                self.kernel_size, self.dilation, self.stride)

            # Reshape weight for this group
            weight_g = self.weight[g*out_group : (g+1)*out_group, :, :]

            # Compute output for this group
            im2col_height = in_group * self.kernel_size
            output_g = torch.einsum('oi, bic -> boc',
                weight_g.view(out_group, im2col_height),
                im2col_g)

            outputs.append(output_g)

        # Concatenate outputs along channel dimension
        output = torch.cat(outputs, dim=1)

        # Add bias
        if self.bias is not None:
            output += self.bias.view(1, -1, 1)

        return output

def get_inputs():
    x = torch.rand(batch_size, in_channels, length).cuda()
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias]