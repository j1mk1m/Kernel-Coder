import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose2d
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void conv_transpose2d_cuda_forward(
    const float* input,
    const float* weight,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int input_h,
    int input_w,
    int output_h,
    int output_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size * out_channels * output_h * output_w)
        return;

    int n = idx / (out_channels * output_h * output_w);
    int remainder = idx % (out_channels * output_h * output_w);

    int c_out = remainder / (output_h * output_w);
    remainder %= (output_h * output_w);

    int y = remainder / output_w;
    int x = remainder % output_w;

    // Determine group and within-group indices
    int out_channels_per_group = out_channels / groups;
    int g = c_out / out_channels_per_group;
    int out_channel_in_group = c_out % out_channels_per_group;

    int in_channels_per_group = in_channels / groups;
    int start_in = g * in_channels_per_group;

    float sum = 0.0f;

    for (int in_c_in_group = 0; in_c_in_group < in_channels_per_group; ++in_c_in_group) {
        int in_c_global = start_in + in_c_in_group;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Compute input coordinates
                int iy = (y - kh * dilation_h + padding_h) / stride_h;
                int ix = (x - kw * dilation_w + padding_w) / stride_w;

                // Check if within input bounds
                if (iy < 0 || iy >= input_h || ix < 0 || ix >= input_w)
                    continue;

                // Calculate weight index
                int weight_offset = in_c_global * out_channels_per_group * kernel_h * kernel_w +
                                    out_channel_in_group * kernel_h * kernel_w +
                                    kh * kernel_w + kw;

                float w = weight[weight_offset];

                // Calculate input index
                int input_offset = n * in_channels * input_h * input_w +
                                   in_c_global * input_h * input_w +
                                   iy * input_w + ix;

                sum += w * input[input_offset];
            }
        }
    }

    // Write output
    int output_offset = n * out_channels * output_h * output_w +
                        c_out * output_h * output_w +
                        y * output_w + x;

    output[output_offset] = sum;
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups) {

    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_h = input.size(2);
    auto input_w = input.size(3);

    auto out_channels = weight.size(0) * weight.size(1); // in_channels, out_channels_per_group, ...
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);

    // Compute output dimensions
    int output_h = (input_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1;
    int output_w = (input_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1;

    auto output = torch::empty({batch_size, out_channels, output_h, output_w}, input.options());

    const int threads = THREADS_PER_BLOCK;
    int total_elements = batch_size * out_channels * output_h * output_w;
    int blocks = (total_elements + threads - 1) / threads;

    conv_transpose2d_cuda_forward<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        input_h,
        input_w,
        output_h,
        output_w
    );

    cudaDeviceSynchronize();
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
);
"""

conv_transpose2d = load_inline(
    name="conv_transpose2d",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_forward"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        # Initialize weights (matching PyTorch's default initialization)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # Reshape and permute the weight to match the expected input format of the kernel
        group_size = self.groups
        out_per_group = self.weight.size(0) // group_size
        in_per_group = self.weight.size(1)
        kernel_h, kernel_w = self.kernel_size

        # Reshape to (groups, out_per_group, in_per_group, kh, kw)
        # Then permute to (groups, in_per_group, out_per_group, kh, kw)
        weight_reshaped = self.weight.view(group_size, out_per_group, in_per_group, kernel_h, kernel_w)
        weight_reshaped = weight_reshaped.permute(0, 2, 1, 3, 4)  # groups, in_per, out_per, kh, kw
        weight_final = weight_reshaped.contiguous().view(
            group_size * in_per_group, out_per_group, kernel_h, kernel_w
        )

        return conv_transpose2d.conv_transpose2d_forward(
            x,
            weight_final,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups
        )